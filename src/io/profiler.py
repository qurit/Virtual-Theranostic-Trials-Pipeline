"""
Pipeline profiling — CPU and RAM sampling per stage.

Enabled only when the pipeline is launched with ``--profile``.  A background
daemon thread samples system-wide CPU % and process RSS every
``interval_s`` seconds (default 2 s).  On pipeline completion, results are
written to ``profiling_CT_<index>.json`` in the CT output folder.

Overhead is negligible: the thread sleeps between two lightweight psutil
syscalls and never acquires any pipeline lock.

JSON schema
-----------
{
  "pipeline_total_elapsed_s": float,
  "config": { ... full pipeline config ... },
  "stages": [
    {
      "name":            str,
      "elapsed_s":       float,
      "cpu_pct_samples": [float, ...],   // system-wide, sampled every interval_s
      "ram_mb_samples":  [float, ...]    // process RSS in MiB
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
from typing import Any, Dict, List, Optional

import psutil


class StageProfiler:
    """
    Lightweight per-stage CPU + RAM profiler.

    Usage
    -----
    profiler = StageProfiler()
    profiler.start_stage("segmentation")
    ... run stage ...
    profiler.end_stage()
    profiler.save(path, pipeline_elapsed_s, config)
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        self._interval = interval_s
        self._stages: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def start_stage(self, name: str) -> None:
        """Begin sampling for a new stage.  Ends any in-progress stage first."""
        self._stop_sampling()
        self._current = {
            "name": name,
            "_start": time.perf_counter(),
            "cpu_pct_samples": [],
            "ram_mb_samples": [],
        }
        self._active = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def end_stage(self) -> None:
        """Stop sampling and record elapsed time for the current stage."""
        self._stop_sampling()
        if self._current is not None:
            elapsed = time.perf_counter() - self._current.pop("_start")
            self._current["elapsed_s"] = round(elapsed, 3)
            self._stages.append(self._current)
            self._current = None

    def save(self, output_path: str, pipeline_elapsed_s: float, config: Dict[str, Any]) -> None:
        """Write profiling results to *output_path* as JSON."""
        payload = {
            "pipeline_total_elapsed_s": round(pipeline_elapsed_s, 3),
            "config": config,
            "stages": self._stages,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _stop_sampling(self) -> None:
        self._active = False
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None

    def _sample_loop(self) -> None:
        proc = psutil.Process(os.getpid())
        # Prime the CPU % counter (first call always returns 0.0).
        psutil.cpu_percent(interval=None)

        while self._active:
            time.sleep(self._interval)
            if not self._active:
                break
            cpu = round(psutil.cpu_percent(interval=None), 1)
            try:
                ram = round(proc.memory_info().rss / (1024 ** 2), 1)
            except psutil.NoSuchProcess:
                ram = 0.0
            with self._lock:
                if self._current is not None:
                    self._current["cpu_pct_samples"].append(cpu)
                    self._current["ram_mb_samples"].append(ram)
