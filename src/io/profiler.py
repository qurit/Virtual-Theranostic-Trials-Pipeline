"""
Pipeline profiling — per-stage CPU and RAM sampling.

Profiling is enabled only when the pipeline is launched with ``--profile``.
Each stage gets its own lightweight background sampler that records:

- pipeline CPU usage across the current Python process and its live children
- pipeline RSS memory across the same process tree
- whole-machine CPU usage for context

The sampler is event-driven, so ending a stage does not block for the remainder
of the sleep interval. This avoids inflating short stages by ~``interval_s``.

JSON schema
-----------
{
  "pipeline_total_elapsed_s": float,
  "sample_interval_s": float,
  "logical_cpu_count": int,
  "config": { ... full pipeline config ... },
  "stages": [
    {
      "name":                   str,
      "elapsed_s":              float,
      "sample_times_s":         [float, ...],
      "cpu_pct_samples":        [float, ...],
      "ram_mb_samples":         [float, ...],
      "system_cpu_pct_samples": [float, ...],
      "process_count_samples":  [int, ...]
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import os
import threading
import time
import argparse
from typing import Any, Dict, List, Optional

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised only when psutil is absent
    psutil = None  # type: ignore[assignment]


class StageProfiler:
    """
    Lightweight per-stage CPU + RAM profiler.

    Usage
    -----
    profiler = StageProfiler(interval_s=2.0)
    profiler.start_stage("segmentation")
    ... run stage ...
    stage_record = profiler.end_stage()
    profiler.save(path, pipeline_elapsed_s, config)
    """

    DEFAULT_INTERVAL_S = 2.0
    MIN_INTERVAL_S = 0.1
    MAX_INTERVAL_S = 3.0

    def __init__(self, interval_s: float = 2.0) -> None:
        if psutil is None:
            raise RuntimeError(
                "Profiling requires the optional 'psutil' package. "
                "Install it or run the pipeline without --profile."
            )
        interval_s = validate_profile_interval_s(interval_s)

        self._interval = interval_s
        self._psutil = psutil
        self._root_proc = self._psutil.Process(os.getpid())
        self._cpu_count = self._psutil.cpu_count(logical=True) or 1
        self._stages: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._autosave_path: Optional[str] = None
        self._autosave_config: Optional[Dict[str, Any]] = None
        self._autosave_start: float = 0.0
        self._proc_cache: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def configure_autosave(self, path: str, config: Dict[str, Any], pipeline_start: float) -> None:
        """Save profiling data to *path* after every stage completes."""
        self._autosave_path = path
        self._autosave_config = config
        self._autosave_start = pipeline_start

    def start_stage(self, name: str) -> None:
        """Begin sampling for a new stage.  Ends any in-progress stage first."""
        self._stop_sampling()
        self._proc_cache = {self._root_proc.pid: self._root_proc}
        self._current = {
            "name": name,
            "_start": time.perf_counter(),
            "sample_times_s": [],
            "cpu_pct_samples": [],
            "ram_mb_samples": [],
            "system_cpu_pct_samples": [],
            "process_count_samples": [],
        }
        self._prime_cpu_counters()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_loop,
            args=(self._current,),
            daemon=True,
        )
        self._thread.start()

    def cancel_stage(self) -> None:
        """Stop sampling and discard the current stage — called when a stage is fully skipped."""
        self._stop_sampling()
        self._current = None

    def end_stage(self) -> Optional[Dict[str, Any]]:
        """Stop sampling, record elapsed time, autosave if configured, and return the stage record."""
        if self._current is None:
            return None

        current = self._current
        self._stop_sampling()
        self._capture_sample(current)

        elapsed = time.perf_counter() - current.pop("_start")
        current["elapsed_s"] = round(elapsed, 3)
        self._stages.append(current)
        self._current = None
        if self._autosave_path is not None:
            pipeline_elapsed = time.perf_counter() - self._autosave_start
            self.save(self._autosave_path, pipeline_elapsed, self._autosave_config or {})
        return current

    def save(self, output_path: str, pipeline_elapsed_s: float, config: Dict[str, Any]) -> None:
        """Write profiling results to *output_path* as JSON."""
        payload = {
            "pipeline_total_elapsed_s": round(pipeline_elapsed_s, 3),
            "sample_interval_s": self._interval,
            "logical_cpu_count": self._cpu_count,
            "config": config,
            "stages": self._stages,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _stop_sampling(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _psutil_errors(self) -> tuple[type[BaseException], ...]:
        zombie = getattr(self._psutil, "ZombieProcess", self._psutil.NoSuchProcess)
        return (
            self._psutil.NoSuchProcess,
            self._psutil.AccessDenied,
            zombie,
            ProcessLookupError,
        )

    def _iter_process_tree(self) -> List[Any]:
        # Reuse cached Process objects so cpu_percent() history accumulates across
        # samples. children() returns new objects each call — always 0.0 on first use.
        live_pids: Dict[int, Any] = {self._root_proc.pid: self._root_proc}
        try:
            for child in self._root_proc.children(recursive=True):
                pid = child.pid
                if pid not in self._proc_cache:
                    self._proc_cache[pid] = child
                    try:
                        child.cpu_percent(interval=None)  # prime — first call always 0
                    except self._psutil_errors():
                        pass
                live_pids[pid] = self._proc_cache[pid]
        except self._psutil_errors():
            pass
        # Drop dead processes from cache
        for pid in list(self._proc_cache):
            if pid not in live_pids:
                del self._proc_cache[pid]
        return list(live_pids.values())

    def _prime_cpu_counters(self) -> None:
        self._psutil.cpu_percent(interval=None)
        for proc in list(self._proc_cache.values()):
            try:
                proc.cpu_percent(interval=None)
            except self._psutil_errors():
                continue

    def _capture_sample(self, stage: Optional[Dict[str, Any]] = None) -> None:
        stage = stage or self._current
        if stage is None:
            return

        sample_time = round(time.perf_counter() - stage["_start"], 3)
        system_cpu = round(self._psutil.cpu_percent(interval=None), 1)

        tree_cpu = 0.0
        tree_rss_bytes = 0
        process_count = 0
        for proc in self._iter_process_tree():
            try:
                with proc.oneshot():
                    tree_cpu += proc.cpu_percent(interval=None)
                    tree_rss_bytes += proc.memory_info().rss
                process_count += 1
            except self._psutil_errors():
                continue

        tree_cpu = round(tree_cpu, 1)
        tree_ram_mb = round(tree_rss_bytes / (1024 ** 2), 1)

        with self._lock:
            if self._current is not stage:
                return
            stage["sample_times_s"].append(sample_time)
            stage["cpu_pct_samples"].append(tree_cpu)
            stage["ram_mb_samples"].append(tree_ram_mb)
            stage["system_cpu_pct_samples"].append(system_cpu)
            stage["process_count_samples"].append(process_count)

    def _sample_loop(self, stage: Dict[str, Any]) -> None:
        while not self._stop_event.wait(self._interval):
            self._capture_sample(stage)


def validate_profile_interval_s(interval_s: float) -> float:
    """Validate and return a profiler sampling interval in seconds."""
    value = float(interval_s)
    if not (StageProfiler.MIN_INTERVAL_S <= value <= StageProfiler.MAX_INTERVAL_S):
        raise ValueError(
            f"interval_s must be between {StageProfiler.MIN_INTERVAL_S} and "
            f"{StageProfiler.MAX_INTERVAL_S} seconds, got {interval_s!r}"
        )
    return value


def parse_profile_interval_arg(raw: str) -> float:
    """Argparse-compatible parser for profiler sampling interval."""
    try:
        return validate_profile_interval_s(float(raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
