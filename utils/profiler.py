from __future__ import annotations

import time
from contextlib import ContextDecorator
from typing import Optional

import torch


class Profiler(ContextDecorator):
    """Simple context manager for timing code blocks and tracking GPU memory."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_file: Optional[str] = None,
        track_gpu: bool = False,
        batches: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.log_file = log_file
        self.track_gpu = track_gpu
        self.batches = batches
        self.start_time: float | None = None
        self.start_mem: int | None = None

    def __enter__(self) -> "Profiler":
        if self.enabled:
            if self.track_gpu and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.start_mem = torch.cuda.memory_allocated()
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        end_time = time.perf_counter()
        elapsed = end_time - (self.start_time or end_time)
        parts = [f"Elapsed {elapsed:.4f}s"]
        if self.batches is not None and elapsed > 0:
            throughput = self.batches / elapsed
            parts.append(f"Throughput {throughput:.2f} batches/s")
        if self.track_gpu and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() - (self.start_mem or 0)
            parts.append(f"GPU peak {peak / 1024 ** 2:.2f} MiB")
        msg = " | ".join(parts)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        else:
            print(msg)
        return False
