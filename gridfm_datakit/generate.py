"""Main data generation module for gridfm_datakit."""

import gc
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import queue 
import signal
from datetime import datetime
from multiprocessing import Manager
from typing import Any, Dict, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

import gridfm_datakit.powsybl as powsybl
from gridfm_datakit.network import (
    Network,
    get_pglib_file_path,
    load_net_from_file,
    load_net_from_pglib,
)
from gridfm_datakit.perturbations.load_perturbation import (
    load_scenarios_to_df,
    reconstruct_scenarios_from_df,
    plot_load_scenarios_combined,
)
from gridfm_datakit.process.process_network import (
    init_julia,
    process_scenario_chunk,
    process_scenario_opf_mode,
    process_scenario_pf_mode,
)
from gridfm_datakit.save import (
    save_node_edge_data,
)
from gridfm_datakit.utils.param_handler import (
    NestedNamespace,
    get_load_scenario_generator,
    initialize_admittance_generator,
    initialize_generation_generator,
    initialize_topology_generator,
)
from gridfm_datakit.utils.random_seed import custom_seed
from gridfm_datakit.utils import profiler
from gridfm_datakit.utils.utils import Tee, write_ram_usage_distributed

@profiler.profile()
def _setup_environment(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Tuple[NestedNamespace, str, Dict[str, str], int]:
    """Setup the environment for data generation.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)

    Returns:
        Tuple of (args, base_path, file_paths, seed)
    """
    # Load config from file if a path is provided
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    # Convert dict to NestedNamespace if needed
    if isinstance(config, dict):
        args = NestedNamespace(**config)
    else:
        args = config

        # Set global seed if provided, otherwise generate a unique seed for this generation
    if (
        hasattr(args.settings, "seed")
        and args.settings.seed is not None
        and args.settings.seed != ""
    ):
        seed = args.settings.seed
        print(f"Global random seed set to: {seed}")

    else:
        # Generate a unique seed for non-reproducible but independent scenarios
        # This ensures scenarios are i.i.d. within a run, but different across runs
        import secrets

        seed = secrets.randbelow(50_000)
        # chunk_seed = seed * 20000 + start_idx + 1 < 2^31 - 1
        # seed < (2,147,483,647 - n_scenarios) / 20,000 ~= 100_000 so taking 50_000 to be safe
        print(f"No seed provided. Using seed={seed}")

    # Resolve and validate the network reader.
    #
    # reader controls HOW the network file is parsed (independent of pf_solver).
    # source controls WHERE to get the file: 'pglib' (download) or 'file' (local).
    reader = getattr(args.network, "reader", "native")
    if reader not in ("native", "powsybl"):
        raise ValueError(
            f"network.reader must be 'native' or 'powsybl', got {reader!r}",
        )
    args.network.reader = reader

    # Resolve and validate the PF solver setting.
    #
    # pf_solver controls which engine is used to solve the power flow equations
    # in PF mode.  It is completely independent of network.source/reader.
    #
    # OPF is always solved by PowerModels (Julia) regardless of this setting.
    # In OPF mode the value is read and stored on args but is never consulted
    # during execution — it is kept here purely for consistency and logging.
    pf_solver = getattr(args.settings, "pf_solver", "powermodel")
    if pf_solver not in ("powermodel", "powsybl"):
        raise ValueError(
            f"settings.pf_solver must be 'powermodel' or 'powsybl', got {pf_solver!r}",
        )
    args.settings.pf_solver = pf_solver

    # Setup output directory
    base_path = os.path.join(args.settings.data_dir, args.network.name, "raw")
    if os.path.exists(base_path) and args.settings.overwrite:
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)

    # Setup solver logs directory under data_dir/solver_log
    solver_log_dir = (
        os.path.join(base_path, "solver_log")
        if args.settings.enable_solver_logs
        else None
    )
    os.makedirs(solver_log_dir, exist_ok=True) if solver_log_dir is not None else None

    # Enable profiling if requested. This must happen before any worker Pool is
    # created so the configuration is inherited by spawned workers (see
    # gridfm_datakit.utils.profiler). Defaults to off when the key is absent.
    if getattr(args.settings, "profiler", False):
        profile_dir = os.path.join(base_path, "profile")
        profiler.enable_profiler(profile_dir, is_main=True)
        print(f"Profiler enabled. Report will be written to {profile_dir}")

    # Setup file paths
    file_paths = {
        "tqdm_log": os.path.join(base_path, "tqdm.log"),
        "error_log": os.path.join(base_path, "error.log"),
        "args_log": os.path.join(base_path, "args.log"),
        "solver_log_dir": solver_log_dir,
        "bus_data": os.path.join(base_path, "bus_data.parquet"),
        "branch_data": os.path.join(base_path, "branch_data.parquet"),
        "gen_data": os.path.join(base_path, "gen_data.parquet"),
        "y_bus_data": os.path.join(base_path, "y_bus_data.parquet"),
        "runtime_data": os.path.join(base_path, "runtime_data.parquet"),
        "scenarios": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.parquet",
        ),
        "scenarios_plot": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.html",
        ),
        "scenarios_log": os.path.join(
            base_path,
            f"scenarios_{args.load.generator}.log",
        ),
    }

    # Initialize logs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for log_file in [
        file_paths["tqdm_log"],
        file_paths["error_log"],
        file_paths["scenarios_log"],
        file_paths["args_log"],
    ]:
        with open(log_file, "a") as f:
            f.write(f"\nNew generation started at {timestamp}\n")
            if log_file == file_paths["args_log"]:
                yaml.safe_dump(args.to_dict(), f)

    return args, base_path, file_paths, seed


@profiler.profile()
def _prepare_network_and_scenarios(
    args: NestedNamespace,
    file_paths: Dict[str, str],
    seed: int,
) -> Tuple[Network, np.ndarray, Dict[str, Any]]:
    """Prepare the network and generate load scenarios.

    Args:
        args: Configuration object
        file_paths: Dictionary of file paths
        seed: Global random seed for reproducibility.

    Returns:
        Tuple of (network, scenarios)
    """
    meta = {}
    reader = args.network.reader  # already validated in _setup_environment

    if args.network.source == "pglib":
        if reader == "powsybl":
            network_path = get_pglib_file_path(args.network.name)
            loaded_net = powsybl.load_net(network_path)
            meta["pp_net"] = loaded_net.pp_net
            meta["network_path"] = network_path
            meta["mapping_p2g"] = loaded_net.mapping_p2g
            net = loaded_net.gfm_net
        else:
            net = load_net_from_pglib(args.network.name)
    elif args.network.source == "file":
        if reader == "powsybl":
            network_path = (
                args.network.file
                if getattr(args.network, "file", None)
                else os.path.join(args.network.network_dir, args.network.name) + ".m"
            )
            loaded_net = powsybl.load_net(network_path)
            meta["pp_net"] = loaded_net.pp_net
            meta["network_path"] = network_path
            meta["mapping_p2g"] = loaded_net.mapping_p2g
            net = loaded_net.gfm_net
        else:
            net = load_net_from_file(
                os.path.join(args.network.network_dir, args.network.name) + ".m",
            )
    else:
        raise ValueError(
            f"network.source must be 'pglib' or 'file', got {args.network.source!r}",
        )

    read_scenarios = True # connect to config
    if read_scenarios:
        # 1. Load the parquet
        #scenarios_df = pd.read_parquet("/home/atif/gridfm-datakit/scripts/large_grids/data_case118_baseline/pf/case118_ieee/raw/scenarios_agg_load_profile.parquet")
        scenarios_df = pd.read_parquet("/home/atif/gridfm-datakit/scripts/large_grids/scenarios_df/opf/case19402/scenarios_agg_load_profile.parquet")
        
        # 2. Infer dimensions
        # Total rows = n_loads * n_scenarios
        # We can get n_scenarios from the max value of 'load_scenario' + 1
        # We can get n_loads from the max value of 'load' + 1
        n_scenarios = scenarios_df["load_scenario"].max() + 1
        n_loads = scenarios_df["load"].max() + 1
        
        # 3. Reconstruct
        scenarios = reconstruct_scenarios_from_df(scenarios_df, n_loads, n_scenarios)
        
        # Verify shape
        print(f"Reconstructed shape: {scenarios.shape}") # Should be (n_loads, n_scenarios, 2)
    else:
        # Generate load scenarios
        load_scenario_generator = get_load_scenario_generator(args.load)
        scenarios = load_scenario_generator(
            net,
            args.load.scenarios,
            file_paths["scenarios_log"],
            max_iter=args.settings.max_iter,
            seed=seed,
        )
        scenarios_df = load_scenarios_to_df(scenarios)
        scenarios_df.to_parquet(file_paths["scenarios"], index=False, engine="pyarrow")
    if net.buses.shape[0] <= 100:
        plot_load_scenarios_combined(scenarios_df, file_paths["scenarios_plot"])
    else:
        print("Skipping plot of scenarios for large networks (number of buses > 100)")

    return net, scenarios, meta


@profiler.profile()
def _save_generated_data(
    net: Network,
    processed_data: List,
    file_paths: Dict[str, str],
    base_path: str,
    args: NestedNamespace,
) -> None:
    """Save the generated data to files.

    Args:
        net: Network object
        processed_data: List of processed data arrays
        file_paths: Dictionary of file paths
        base_path: Base output directory
        args: Configuration object
    """
    if len(processed_data) > 0:
        save_node_edge_data(
            net,
            file_paths["bus_data"],
            file_paths["branch_data"],
            file_paths["gen_data"],
            file_paths["y_bus_data"],
            file_paths["runtime_data"],
            processed_data,
            include_dc_res=args.settings.include_dc_res,
        )


@profiler.profile()
def generate_power_flow_data(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Dict[str, str]:
    """Generate power flow data based on the provided configuration using sequential processing.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and perturbation configurations.

    Returns:
        Dictionary with paths to generated artifacts:
        {
            'tqdm_log': progress log file,
            'error_log': error log file,
            'args_log': configuration dump file,
            'bus_data': bus-level features CSV (BUS_COLUMNS),
            'branch_data': branch-level features CSV (BRANCH_COLUMNS),
            'gen_data': generator features CSV (GEN_COLUMNS),
            'y_bus_data': Y-bus nonzero entries CSV,
            'scenarios': load scenarios Parquet,
            'scenarios_plot': load scenarios plot HTML,
            'scenarios_log': load scenario generation log
        }

    Note:
        The function creates output files under {settings.data_dir}/{network.name}/raw/:

        - tqdm.log: Progress tracking
        - error.log: Error messages
        - args.log: Configuration parameters (YAML dump)
        - bus_data.parquet: Bus-level features for each scenario
        - branch_data.parquet: Branch-level features for each scenario
        - gen_data.parquet: Generator features for each scenario
        - y_bus_data.parquet: Nonzero Y-bus entries for each scenario
        - scenarios_{generator}.parquet: Load scenarios (per-element time series)
        - scenarios_{generator}.html: Load scenario plots
        - scenarios_{generator}.log: Load scenario generation notes
    """

    # Setup environment
    args, base_path, file_paths, seed = _setup_environment(config)

    # Prepare network and scenarios
    net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)

    # Initialize topology generator
    topology_generator = initialize_topology_generator(args.topology_perturbation, net)

    # Initialize generation generator
    generation_generator = initialize_generation_generator(
        args.generation_perturbation,
        net,
    )

    # Initialize admittance generator
    admittance_generator = initialize_admittance_generator(
        args.admittance_perturbation,
        net,
    )

    # Extract coinhsl settings from config
    coinhsl_config = getattr(args.settings, "coinhsl", None)
    coinhsl_enabled = False
    coinhsl_linear_solver = ""
    coinhsl_hsllib = ""
    
    if coinhsl_config is not None:
        coinhsl_enabled = getattr(coinhsl_config, "enabled", False)
        if coinhsl_enabled:
            coinhsl_linear_solver = getattr(coinhsl_config, "linear_solver", "ma57")
            coinhsl_hsllib = getattr(coinhsl_config, "hsllib", "/home/atif/packages/coinhsl-2023.11.17/install/lib/x86_64-linux-gnu/libcoinhsl.so")
            if not os.path.exists(coinhsl_hsllib):
                raise FileNotFoundError(f"CoinHSL library not found at: {coinhsl_hsllib}")

    jl = init_julia(
        args.settings.max_iter,
        file_paths["solver_log_dir"],
        coinhsl_enabled=coinhsl_enabled,
        coinhsl_linear_solver=coinhsl_linear_solver,
        coinhsl_hsllib=coinhsl_hsllib,
    )

    processed_data = []

    # Process scenarios sequentially with deterministic seed
    # Use custom_seed to control randomness for reproducibility
    # Limit to first 64 scenarios for actual processing
    n_scenarios_to_process = args.load.scenarios #min(args.load.scenarios, 64)
    with custom_seed(seed + 1):
        with open(file_paths["tqdm_log"], "a") as f:
            with tqdm(
                total=n_scenarios_to_process,
                desc="Processing scenarios",
                file=Tee(sys.stdout, f),
                miniters=5,
            ) as pbar:
                for scenario_index in range(n_scenarios_to_process):
                    # Process the scenario
                    if args.settings.mode == "opf":
                        processed_data = process_scenario_opf_mode(
                            net,
                            scenarios,
                            scenario_index,
                            topology_generator,
                            generation_generator,
                            admittance_generator,
                            processed_data,
                            file_paths["error_log"],
                            args.settings.include_dc_res,
                            jl,
                        )
                    elif args.settings.mode == "pf":
                        processed_data = process_scenario_pf_mode(
                            net,
                            scenarios,
                            scenario_index,
                            topology_generator,
                            generation_generator,
                            admittance_generator,
                            processed_data,
                            file_paths["error_log"],
                            args.settings.include_dc_res,
                            args.settings.pf_fast,
                            args.settings.dcpf_fast,
                            jl,
                            args.settings.pf_solver,
                            meta=meta,
                        )
                    else:
                        raise ValueError("Invalid mode!")

                    pbar.update(1)

    # Save final data
    _save_generated_data(
        net,
        processed_data,
        file_paths,
        base_path,
        args,
    )

    # Merge per-process profiling stats and write the report.
    if profiler.is_enabled():
        report_path = profiler.write_report()
        if report_path is not None:
            print(f"Profiling report written to {report_path}")

    return file_paths

@profiler.profile()
def generate_power_flow_data_distributed(
    config: Union[str, Dict[str, Any], NestedNamespace],
) -> Dict[str, str]:
    """Generate power-flow data based on the provided configuration using distributed processing.
    Each worker processes a contiguous sub-chunk of scenarios.
    If settings.scenario_timeout_sec is set, each individual scenario
    receives that many seconds of wall-clock time. If a scenario exceeds
    the timeout, its worker is killed and the whole pool is recreated.

    Ordinary worker exceptions are reported immediately through the progress queue.

    Args:
        config: Configuration can be provided in three ways:
            1. Path to a YAML config file (str)
            2. Configuration dictionary (Dict)
            3. NestedNamespace object (NestedNamespace)
            The config must include settings, network, load, and perturbation configurations.

    Returns:
        Dictionary with paths to generated artifacts (same as generate_power_flow_data)

    Note:
        The function creates output files under {settings.data_dir}/{network.name}/raw/:

        - tqdm.log: Progress tracking
        - error.log: Error messages
        - args.log: Configuration parameters (YAML dump)
        - bus_data.parquet: Bus-level features for each scenario
        - branch_data.parquet: Branch-level features for each scenario
        - gen_data.parquet: Generator features for each scenario
        - y_bus_data.parquet: Nonzero Y-bus entries for each scenario
        - scenarios_{generator}.parquet: Load scenarios (per-element time series)
        - scenarios_{generator}.html: Load scenario plots
        - scenarios_{generator}.log: Load scenario generation notes
    """
    t0 = time.time()
    # Setup environment
    args, base_path, file_paths, seed = _setup_environment(config)

    print('\n Time for _setup_environment',time.time()-t0,flush=True)
    t0 = time.time()

    # check if mode is valid
    if args.settings.mode not in ["opf", "pf"]:
        raise ValueError("Invalid mode!")

    # read scenario timeout from configuration
    scenario_timeout = getattr(args.settings,"scenario_timeout_sec",None)

    if scenario_timeout is not None:
        scenario_timeout = float(scenario_timeout)

        if scenario_timeout <= 0:
            raise ValueError(
                "settings.scenario_timeout_sec must be "
                "greater than zero or None"
            )

    # Prepare network and scenarios
    net, scenarios, meta = _prepare_network_and_scenarios(args, file_paths, seed)

    print('\n Time for _prepare_network_and_scenarios',time.time()-t0,flush=True)
    t0 = time.time()

    # Initialize topology generator
    topology_generator = initialize_topology_generator(args.topology_perturbation, net)

    print('\n Time for initialize_topology_generator',time.time()-t0,flush=True)
    t0 = time.time()

    # Initialize generation generator
    generation_generator = initialize_generation_generator(
        args.generation_perturbation,
        net,
    )

    print('\n Time for initialize_generation_generator',time.time()-t0,flush=True)
    t0 = time.time()

    # Initialize admittance generator
    admittance_generator = initialize_admittance_generator(
        args.admittance_perturbation,
        net,
    )

    print('\n Time for initialize_admittance_generator',time.time()-t0,flush=True)
    t0 = time.time()

    # Setup multiprocessing
    mp_ctx = multiprocessing.get_context("spawn")
    manager = mp_ctx.Manager()
    progress_queue = manager.Queue()

    # Process scenarios in chunks - limit to first 64 scenarios for actual processing
    # while keeping full scenarios array for interpolation purposes
    n_scenarios_to_process = args.load.scenarios #min(args.load.scenarios, 64)
    large_chunks = np.array_split(
        range(n_scenarios_to_process),
        np.ceil(n_scenarios_to_process / args.settings.large_chunk_size).astype(int),
    )

    # Checkpoint file for resuming from last completed large chunk
    checkpoint_file = os.path.join(base_path, "completed_chunk.txt")
    start_chunk_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            content = f.read().strip()
            if content:
                start_chunk_index = int(content) + 1
                print(f"Resuming from large chunk index {start_chunk_index}")


    try:
        with (
            open(file_paths["tqdm_log"], "a") as f,
            open(file_paths["error_log"], "a") as err_f,
        ):
            with tqdm(
                total=n_scenarios_to_process,
                desc="Processing scenarios",
                file=Tee(sys.stdout, f),
                miniters=5,
            ) as pbar:

                for large_chunk_index, large_chunk in enumerate(
                    large_chunks
                ):
                    # --------------------------------------------------
                    # Already checkpointed.
                    # --------------------------------------------------
                    if large_chunk_index < start_chunk_index:
                        pbar.update(len(large_chunk))
                        continue

                    write_ram_usage_distributed(f)

                    chunk_size = len(large_chunk)

                    scenario_chunks = np.array_split(
                        large_chunk,
                        min(
                            args.settings.num_processes,
                            chunk_size,
                        ),
                    )

                    worker_meta = {
                        k: v
                        for k, v in meta.items()
                        if k != "pp_net"
                    }

                    # --------------------------------------------------
                    # CoinHSL settings
                    # --------------------------------------------------
                    coinhsl_config = getattr(
                        args.settings,
                        "coinhsl",
                        None,
                    )

                    coinhsl_enabled = False
                    coinhsl_linear_solver = ""
                    coinhsl_hsllib = ""

                    if coinhsl_config is not None:
                        coinhsl_enabled = getattr(
                            coinhsl_config,
                            "enabled",
                            False,
                        )

                        if coinhsl_enabled:
                            coinhsl_linear_solver = getattr(
                                coinhsl_config,
                                "linear_solver",
                                "ma57",
                            )

                            coinhsl_hsllib = getattr(
                                coinhsl_config,
                                "hsllib",
                                (
                                    "/home/atif/packages/"
                                    "coinhsl-2023.11.17/install/lib/"
                                    "x86_64-linux-gnu/libcoinhsl.so"
                                ),
                            )

                            if not os.path.exists(
                                coinhsl_hsllib
                            ):
                                raise FileNotFoundError(
                                    "CoinHSL library not found at: "
                                    f"{coinhsl_hsllib}"
                                )

                    # --------------------------------------------------
                    # IMPORTANT:
                    #
                    # This survives retries of THIS large chunk.
                    #
                    # Any scenario added here will be skipped by
                    # process_scenario_chunk on all subsequent attempts.
                    # --------------------------------------------------
                    skipped_scenarios = set()

                    processed_data = None

                    # ==================================================
                    # Retry current large chunk until all non-skipped
                    # scenarios complete successfully.
                    # ==================================================
                    while processed_data is None:

                        # ----------------------------------------------
                        # Rebuild tasks every retry because
                        # skipped_scenarios may have changed.
                        #
                        # Use a regular set copy. It gets pickled when
                        # sent to spawned workers.
                        # ----------------------------------------------
                        skip_snapshot = set(skipped_scenarios)

                        tasks = [
                            (
                                args.settings.mode,
                                int(chunk[0]),
                                int(chunk[-1]) + 1,
                                scenarios,
                                net,
                                progress_queue,
                                topology_generator,
                                generation_generator,
                                admittance_generator,
                                file_paths["error_log"],
                                args.settings.include_dc_res,
                                args.settings.pf_fast,
                                args.settings.dcpf_fast,
                                file_paths["solver_log_dir"],
                                args.settings.max_iter,
                                seed,
                                args.settings.pf_solver,
                                worker_meta,
                                coinhsl_enabled,
                                coinhsl_linear_solver,
                                coinhsl_hsllib,
                                skip_snapshot,
                            )
                            for chunk in scenario_chunks
                        ]

                        pool = mp_ctx.Pool(
                            processes=args.settings.num_processes
                        )

                        pool_closed = False
                        pool_terminated = False

                        # Number of successful scenario completions
                        # reported during THIS attempt.
                        completed_this_attempt = 0

                        try:
                            results = [
                                pool.apply_async(
                                    process_scenario_chunk,
                                    task,
                                )
                                for task in tasks
                            ]

                            timed_out = False
                            timed_out_pid = None
                            timed_out_scenario = None

                            worker_failed = False
                            failed_pid = None
                            failed_scenario = None

                            # pid -> (scenario_index, start_time)
                            active_scenarios = {}

                            # Timed-out scenarios are not expected to emit
                            # "done" during this or future attempts.
                            expected_completions = (
                                chunk_size
                                - len(skipped_scenarios)
                            )

                            # ==========================================
                            # Watch workers/scenarios
                            # ==========================================
                            while (
                                completed_this_attempt
                                < expected_completions
                            ):
                                try:
                                    (
                                        event,
                                        worker_pid,
                                        scenario_index,
                                    ) = progress_queue.get(
                                        timeout=0.5
                                    )

                                    if event == "start":
                                        active_scenarios[
                                            worker_pid
                                        ] = (
                                            scenario_index,
                                            time.monotonic(),
                                        )

                                    elif event == "done":
                                        active_scenarios.pop(
                                            worker_pid,
                                            None,
                                        )

                                        completed_this_attempt += 1
                                        pbar.update(1)

                                    elif event == "error":
                                        active_scenarios.pop(
                                            worker_pid,
                                            None,
                                        )

                                        failed_pid = worker_pid
                                        failed_scenario = (
                                            scenario_index
                                        )
                                        worker_failed = True
                                        break

                                    else:
                                        raise RuntimeError(
                                            "Unknown worker progress "
                                            f"event: {event!r}"
                                        )

                                except queue.Empty:
                                    pass

                                # --------------------------------------
                                # Check every running scenario
                                # independently.
                                # --------------------------------------
                                if scenario_timeout is not None:
                                    now = time.monotonic()

                                    for (
                                        worker_pid,
                                        (
                                            scenario_index,
                                            start_time,
                                        ),
                                    ) in list(
                                        active_scenarios.items()
                                    ):
                                        elapsed = (
                                            now - start_time
                                        )

                                        if (
                                            elapsed
                                            > scenario_timeout
                                        ):
                                            timed_out = True
                                            timed_out_pid = (
                                                worker_pid
                                            )
                                            timed_out_scenario = (
                                                scenario_index
                                            )

                                            message = (
                                                f"Large chunk "
                                                f"{large_chunk_index}: "
                                                f"scenario "
                                                f"{scenario_index} "
                                                f"timed out after "
                                                f"{elapsed:.1f}s "
                                                f"(limit="
                                                f"{scenario_timeout}s, "
                                                f"pid={worker_pid}). "
                                                f"The scenario will be "
                                                f"skipped permanently "
                                                f"for this chunk.\n"
                                            )

                                            print(
                                                message.rstrip(),
                                                flush=True,
                                            )

                                            err_f.write(message)
                                            err_f.flush()

                                            # Kill the worker actually
                                            # running the hung scenario.
                                            try:
                                                os.kill(
                                                    worker_pid,
                                                    signal.SIGKILL,
                                                )
                                            except ProcessLookupError:
                                                pass

                                            break

                                if timed_out or worker_failed:
                                    break

                            # ==========================================
                            # Normal worker exception
                            # ==========================================
                            if worker_failed:
                                message = (
                                    f"Worker pid={failed_pid} "
                                    f"failed"
                                )

                                if failed_scenario is not None:
                                    message += (
                                        " while processing scenario "
                                        f"{failed_scenario}"
                                    )

                                message += ". See error log.\n"

                                err_f.write(message)
                                err_f.flush()

                                pool.terminate()
                                pool.join()

                                pool_terminated = True

                                # This is a real exception, not a
                                # timeout. Do not retry.
                                raise RuntimeError(
                                    message.rstrip()
                                )

                            # ==========================================
                            # Individual scenario timed out
                            # ==========================================
                            if timed_out:
                                # --------------------------------------
                                # THIS is the key behavior:
                                # remember this scenario before retrying.
                                # --------------------------------------
                                skipped_scenarios.add(
                                    timed_out_scenario
                                )

                                pool.terminate()
                                pool.join()

                                pool_terminated = True

                                # --------------------------------------
                                # Drain messages from the aborted pool.
                                # No workers remain after join(), so no
                                # additional old events should arrive.
                                # --------------------------------------
                                while True:
                                    try:
                                        progress_queue.get_nowait()
                                    except queue.Empty:
                                        break

                                # --------------------------------------
                                # All completed work from this aborted
                                # attempt will be recomputed, so remove
                                # it from tqdm.
                                # --------------------------------------
                                if completed_this_attempt:
                                    pbar.n -= (
                                        completed_this_attempt
                                    )
                                    pbar.refresh()

                                # --------------------------------------
                                # The timed-out scenario itself is now
                                # permanently considered handled/skipped.
                                #
                                # It won't be executed again, so count
                                # it once toward overall progress.
                                # --------------------------------------
                                pbar.update(1)

                                continue

                            # ==========================================
                            # All expected non-skipped scenarios emitted
                            # "done". Gather their actual return values.
                            # ==========================================
                            local_results = []

                            for result in results:
                                (
                                    error,
                                    tb,
                                    local_processed_data,
                                ) = result.get()

                                if isinstance(
                                    error,
                                    Exception,
                                ):
                                    print(
                                        "Error in "
                                        "process_scenario_chunk: "
                                        f"{error}"
                                    )

                                    if tb:
                                        print(tb)

                                    raise RuntimeError(
                                        "process_scenario_chunk "
                                        f"failed: {error}"
                                    ) from error

                                if local_processed_data:
                                    local_results.extend(
                                        local_processed_data
                                    )

                            pool.close()
                            pool.join()

                            pool_closed = True

                            processed_data = local_results

                        finally:
                            # ------------------------------------------
                            # Ensure no worker survives an unexpected
                            # exception in the parent.
                            # ------------------------------------------
                            if (
                                not pool_closed
                                and not pool_terminated
                            ):
                                pool.terminate()
                                pool.join()

                    # ==================================================
                    # Large chunk completed.
                    # ==================================================
                    _save_generated_data(
                        net,
                        processed_data,
                        file_paths,
                        base_path,
                        args,
                    )

                    # Log skipped scenarios for visibility.
                    if skipped_scenarios:
                        skipped_sorted = sorted(
                            skipped_scenarios
                        )

                        message = (
                            f"Large chunk {large_chunk_index} "
                            f"completed with timed-out scenarios "
                            f"skipped: {skipped_sorted}\n"
                        )

                        err_f.write(message)
                        err_f.flush()

                    # --------------------------------------------------
                    # Only checkpoint after successfully saving the
                    # completed/non-timed-out data.
                    # --------------------------------------------------
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        dir=base_path,
                        delete=False,
                    ) as tmp:
                        tmp.write(
                            str(large_chunk_index)
                        )
                        tmp_path = tmp.name

                    os.replace(
                        tmp_path,
                        checkpoint_file,
                    )

                    del processed_data
                    gc.collect()

    finally:
        manager.shutdown()

    print(
        "\n Time for data generation",
        time.time() - t0,
        flush=True,
    )

    if profiler.is_enabled():
        report_path = profiler.write_report()

        if report_path is not None:
            print(
                f"Profiling report written to {report_path}"
            )

    return file_paths
