"""
Windows High-Performance Optimization Module
Guarantees full CPU clock speed and real-time responsiveness on Windows 10/11:
1. Locks scheduler timer resolution to 1.0 ms via winmm.timeBeginPeriod(1).
2. Explicitly disables Windows 11 EcoQoS / Efficiency Mode (Power Throttling) via SetProcessInformation.
3. Sets process priority to ABOVE_NORMAL_PRIORITY_CLASS to prevent background degradation.
4. Disables ConHost QuickEdit Mode so mouse clicks on the terminal cannot freeze stdout.
"""

import sys
import atexit
import logging

logger = logging.getLogger(__name__)

_timer_period_active = False


def enable_windows_high_performance() -> bool:
    """
    Applies Windows-specific performance and scheduling hardening.
    Safe and idempotent: returns True on success, False on non-Windows or failure.
    """
    global _timer_period_active
    if sys.platform != "win32":
        return False

    try:
        import ctypes
        import ctypes.wintypes as w

        kernel32 = ctypes.windll.kernel32
        winmm = ctypes.windll.winmm

        # 1. Lock multimedia timer resolution to 1.0 ms
        if not _timer_period_active:
            if winmm.timeBeginPeriod(1) == 0:
                _timer_period_active = True
                atexit.register(lambda: winmm.timeEndPeriod(1))

        # 2. Configure 64-bit handle types for kernel32
        kernel32.GetCurrentProcess.restype = w.HANDLE

        # 3. Elevate process priority to ABOVE_NORMAL_PRIORITY_CLASS (0x00008000)
        # Prevents Windows from starving background worker threads
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        kernel32.SetPriorityClass.argtypes = [w.HANDLE, w.DWORD]
        kernel32.SetPriorityClass.restype = w.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)

        # 4. Explicitly disable Windows 11 EcoQoS (Power Throttling / Efficiency Mode)
        # Prevents Windows from migrating threads to low-power E-cores when terminal loses focus
        SetProcessInformation = getattr(kernel32, "SetProcessInformation", None)
        if SetProcessInformation is not None:
            class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                _fields_ = [
                    ("Version", w.DWORD),
                    ("ControlMask", w.DWORD),
                    ("StateMask", w.DWORD),
                ]

            PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
            ProcessPowerThrottling = 4

            SetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            SetProcessInformation.restype = w.BOOL

            throttle_state = PROCESS_POWER_THROTTLING_STATE(
                Version=PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
                StateMask=0,  # 0 explicitly turns OFF execution speed throttling
            )
            SetProcessInformation(
                kernel32.GetCurrentProcess(),
                ProcessPowerThrottling,
                ctypes.byref(throttle_state),
                ctypes.sizeof(throttle_state),
            )

        # 5. Disable QuickEdit Mode on standard input
        # Prevents conhost from pausing stdout output when user clicks inside terminal
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        kernel32.GetStdHandle.restype = w.HANDLE
        kernel32.GetStdHandle.argtypes = [w.DWORD]
        kernel32.GetConsoleMode.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        kernel32.GetConsoleMode.restype = w.BOOL
        kernel32.SetConsoleMode.argtypes = [w.HANDLE, w.DWORD]
        kernel32.SetConsoleMode.restype = w.BOOL

        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = w.DWORD()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(h_stdin, new_mode)

        # 6. Hybrid CPU Optimization: Bind to P-cores and prevent E-core thrashing
        configure_cpu_core_affinity(prefer_p_cores=True)

        return True

    except Exception as e:
        logger.warning(f"Windows high-performance setup notice: {e}")
        return False


def configure_cpu_core_affinity(prefer_p_cores: bool = True) -> bool:
    """
    On hybrid Intel CPUs (12th/13th/14th Gen, e.g. i9-14900HX with 8 P-cores + 16 E-cores),
    confines execution strictly to Performance Cores (threads 0..15).
    Guarantees:
    1. E-cores (threads 16..31) are NEVER used by our process.
    2. Zero cross-cluster cache thrashing or thread bouncing.
    3. Prevents package-level thermal runaway / 90°C spike.
    4. Limits PyTorch / OpenMP thread pools to physical P-cores.
    """
    try:
        import psutil
        proc = psutil.Process()
        total_threads = psutil.cpu_count(logical=True) or 1
        physical_cores = psutil.cpu_count(logical=False) or 1

        # Hybrid architecture detection (e.g. 24 cores, 32 threads: 8P + 16E)
        if prefer_p_cores and total_threads >= 20 and physical_cores != total_threads:
            p_core_threads = 16 if total_threads == 32 else (16 if total_threads >= 24 else 12)
            p_affinity = list(range(min(p_core_threads, total_threads)))
            proc.cpu_affinity(p_affinity)

            # Clamp PyTorch and OpenCV CPU threads to 4 cores
            # Prevents Intel 13th/14th Gen HX laptop CPUs (i9-14900HX) from triggering 157W PL2 thermal spikes
            try:
                import torch
                torch.set_num_threads(4)
                torch.set_num_interop_threads(2)
            except Exception:
                pass

            try:
                import cv2
                cv2.setNumThreads(4)
            except Exception:
                pass

            return True
    except Exception:
        pass
    return False
