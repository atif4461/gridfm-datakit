"""Lightweight, opt-in profiler for gridfm-datakit.

Targeted timing of the functions, routines and IO that dominate power-flow
data generation. Enabled with ``settings.profiler: true`` in the YAML config.

Design notes
------------
The heavy work runs inside spawned worker processes (see
``generate_power_flow_data_distributed``), so a profiler living only in the
main process would see almost nothing. Configuration is therefore propagated
to workers through environment variables, which are inherited across the
``spawn`` start method.

Each process accumulates per-function call counts and elapsed wall time in
memory. Worker processes flush their stats to a unique JSON file when they
exit; the main process then merges every worker file with its own in-memory
stats and writes a single human-readable ``profile_report.txt``.

Instrumentation is done with the :func:`profile` decorator and the
:func:`profile_block` context manager. Both are no-ops (a single boolean
check) when profiling is disabled, so they are safe to leave in the code.

Julia solver calls are timed as a black box: the wall time of the blocking
``jl.run_*`` call is recorded under a ``julia.*`` label. We do not look
inside the Julia/Ipopt kernel.
"""

import atexit
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, List, Optional

# Environment variables used to propagate configuration to spawned workers.
_ENV_ENABLED = "GRIDFM_PROFILE"
_ENV_DIR = "GRIDFM_PROFILE_DIR"

_enabled = False
_output_dir: Optional[str] = None
# Unique per process instance so worker stat files never collide (pids can be
# reused across the per-chunk process pools).
_proc_uuid = uuid.uuid4().hex
_lock = threading.Lock()
_thread_local = threading.local()

# label -> [call_count, total_time, self_time]
_stats: Dict[str, List[float]] = {}
_atexit_registered = False


def _get_stack() -> List[float]:
    """Per-thread stack of accumulated child time, for self-time bookkeeping."""
    try:
        return _thread_local.stack
    except AttributeError:
        _thread_local.stack = []
        return _thread_local.stack


def is_enabled() -> bool:
    """Return True if profiling is active in this process."""
    return _enabled


def enable_profiler(output_dir: str, is_main: bool = True) -> None:
    """Turn on profiling in this process and propagate it to child processes.

    Args:
        output_dir: Directory where per-process stat files and the final
            report are written.
        is_main: True for the main process. The main process clears stale
            stat files (so a re-run does not merge old data) and does NOT
            register an ``atexit`` dump, because it writes the merged report
            explicitly via :func:`write_report` before the interpreter exits.
    """
    global _enabled, _output_dir
    _enabled = True
    _output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Propagate to spawned workers via the environment.
    os.environ[_ENV_ENABLED] = "1"
    os.environ[_ENV_DIR] = output_dir

    if is_main:
        # Start from a clean slate so a previous run's worker files are not
        # merged into this run's report.
        for fname in os.listdir(output_dir):
            if fname.startswith("profile_") and fname.endswith(".json"):
                try:
                    os.remove(os.path.join(output_dir, fname))
                except OSError:
                    pass
    else:
        _register_dump()


def _init_from_env() -> None:
    """Pick up configuration inherited from a parent process (spawn workers)."""
    global _enabled, _output_dir
    if os.environ.get(_ENV_ENABLED) == "1":
        _enabled = True
        _output_dir = os.environ.get(_ENV_DIR)
        _register_dump()


def _register_dump() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_dump)
        _atexit_registered = True


@contextmanager
def _measure(label: str):
    """Record one timed invocation of ``label`` (call count, total, self)."""
    stack = _get_stack()
    stack.append(0.0)  # accumulator for time spent in profiled children
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        child_time = stack.pop()
        self_time = elapsed - child_time
        if stack:
            # Attribute our full duration to the parent frame's child time.
            stack[-1] += elapsed
        with _lock:
            rec = _stats.get(label)
            if rec is None:
                _stats[label] = [1, elapsed, self_time]
            else:
                rec[0] += 1
                rec[1] += elapsed
                rec[2] += self_time


def profile(name: Optional[str] = None) -> Callable:
    """Decorator that times a function and counts its calls when enabled.

    Args:
        name: Optional label. Defaults to ``module.qualname``.
    """

    def decorator(func: Callable) -> Callable:
        label = name or f"{func.__module__}.{func.__qualname__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return func(*args, **kwargs)
            with _measure(label):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def profile_block(name: str):
    """Time an arbitrary block (a routine or an IO operation) when enabled."""
    if not _enabled:
        yield
        return
    with _measure(name):
        yield


def _dump() -> None:
    """Write this process's accumulated stats to a unique JSON file."""
    if not _enabled or not _output_dir:
        return
    with _lock:
        if not _stats:
            return
        snapshot = {k: list(v) for k, v in _stats.items()}
    path = os.path.join(
        _output_dir,
        f"profile_{os.getpid()}_{_proc_uuid}.json",
    )
    try:
        with open(path, "w") as f:
            json.dump(snapshot, f)
    except OSError:
        pass


def write_report(output_dir: Optional[str] = None) -> Optional[str]:
    """Merge this process's stats with worker stat files and write a report.

    Args:
        output_dir: Directory holding the per-process JSON files. Defaults to
            the directory passed to :func:`enable_profiler`.

    Returns:
        Path to the written ``profile_report.txt``, or None if profiling was
        never enabled / no directory is known.
    """
    out = output_dir or _output_dir
    if not out:
        return None

    # label -> [call_count, total_time, self_time]
    merged: Dict[str, List[float]] = {}

    def _add(label: str, count: float, total: float, self_t: float) -> None:
        rec = merged.get(label)
        if rec is None:
            merged[label] = [count, total, self_t]
        else:
            rec[0] += count
            rec[1] += total
            rec[2] += self_t

    # In-memory stats from the current (main) process.
    with _lock:
        for label, (count, total, self_t) in _stats.items():
            _add(label, count, total, self_t)

    # Stats flushed by worker processes.
    if os.path.isdir(out):
        for fname in sorted(os.listdir(out)):
            if not (fname.startswith("profile_") and fname.endswith(".json")):
                continue
            try:
                with open(os.path.join(out, fname)) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            for label, vals in data.items():
                _add(label, vals[0], vals[1], vals[2])

    # Sort by total (cumulative) time descending.
    rows = sorted(merged.items(), key=lambda kv: kv[1][1], reverse=True)

    lines = [
        "gridfm-datakit profiling report",
        "=" * 100,
        (
            "Aggregated across the main process and all worker processes.\n"
            "  total(s) : cumulative wall time in the function, including "
            "profiled callees.\n"
            "  self(s)  : total minus time spent in other profiled functions "
            "it called.\n"
            "  avg(ms)  : total / calls, in milliseconds.\n"
            "Julia solver calls (julia.*) are timed as black-box kernel "
            "runtime; IO is labelled io.*."
        ),
        "",
    ]
    header = (
        f"{'function / block':<64}{'calls':>10}"
        f"{'total(s)':>14}{'self(s)':>14}{'avg(ms)':>13}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for label, (count, total, self_t) in rows:
        icount = int(count)
        avg_ms = (total / icount * 1000.0) if icount else 0.0
        lines.append(
            f"{label[:64]:<64}{icount:>10d}"
            f"{total:>14.4f}{self_t:>14.4f}{avg_ms:>13.3f}",
        )
    lines.append("")

    report_path = os.path.join(out, "profile_report.txt")
    try:
        with open(report_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        return None
    return report_path


# When imported inside a spawned worker, pick up configuration the parent set.
_init_from_env()
