"""Peak-RSS guard for the offline research scripts (the VPS has 909 MB RAM and no swap).

peak_rss_mb() reads the process's peak resident set size: resource.getrusage on Linux / macOS,
the peak working set (ctypes, GetProcessMemoryInfo) on Windows. RssGuard.check() compares it with a
limit between markets and stages and raises MemoryLimitExceeded before the OS would kill the
process; stage() also logs it at INFO.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

logger = logging.getLogger("research.memory")


class MemoryLimitExceeded(RuntimeError):
    """Peak RSS went over --max-rss-mb: the run stops cleanly instead of being OOM-killed."""

    def __init__(self, stage: str, peak_mb: float, limit_mb: float) -> None:
        super().__init__(f"peak RSS {peak_mb:.0f} MB > limit {limit_mb:.0f} MB at stage '{stage}'")
        self.stage, self.peak_mb, self.limit_mb = stage, peak_mb, limit_mb


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB, or None when it cannot be read."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KB on Linux
    except ImportError:
        pass
    try:  # Windows
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit pseudo-handle: types must be explicit
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except (AttributeError, OSError, ImportError):
        pass
    return None


class RssGuard:
    """check() between markets (silent), stage() at stage boundaries (logged at INFO); both raise
    MemoryLimitExceeded when the peak RSS is over limit_mb. An unreadable RSS disables the guard
    with one WARNING."""

    def __init__(self, limit_mb: float, measure: Callable[[], float | None] = peak_rss_mb) -> None:
        self.limit_mb = limit_mb
        self._measure = measure
        self._warned = False
        self.peak_mb: float | None = None

    def check(self, stage: str) -> float | None:
        peak = self._measure()
        if peak is None:
            if not self._warned:
                logger.warning("Peak RSS unreadable on this platform: memory guard disabled")
                self._warned = True
            return None
        self.peak_mb = peak if self.peak_mb is None else max(self.peak_mb, peak)
        if peak > self.limit_mb:
            raise MemoryLimitExceeded(stage, peak, self.limit_mb)
        return peak

    def stage(self, stage: str) -> float | None:
        peak = self.check(stage)
        logger.info("Peak RSS", extra={"stage": stage, "peak_rss_mb": None if peak is None else round(peak, 1), "limit_mb": self.limit_mb})
        return peak
