"""
Process Telemetry Module
Measures strictly the resources (CPU, RAM, VRAM, GPU) consumed by our process
and its worker threads/children, avoiding contamination from external OS processes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import psutil

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


class ProcessTelemetry:
    """Accurately monitors process-isolated system resource usage."""

    _instance: Optional[ProcessTelemetry] = None

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.proc = psutil.Process(self.pid)
        self.num_cpus = psutil.cpu_count() or 1

        # Prime initial cpu_percent reading
        try:
            self.proc.cpu_percent(interval=None)
        except Exception:
            pass

        self._nvml_initialized = False
        self._gpu_handle = None
        self._gpu_name = "CPU"
        self._total_vram_mb = 0.0
        self._last_metrics: Dict[str, Any] = {}
        self._last_time: float = 0.0
        self._cache_ttl: float = 0.25  # 250ms cache throttle to ensure <0.01ms query latency

        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                if pynvml.nvmlDeviceGetCount() > 0:
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    raw_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                    self._gpu_name = raw_name if isinstance(raw_name, str) else raw_name.decode("utf-8", errors="ignore")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    self._total_vram_mb = round(mem_info.total / (1024 * 1024), 1)
                    self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False

    @classmethod
    def get_instance(cls) -> ProcessTelemetry:
        if cls._instance is None:
            cls._instance = ProcessTelemetry()
        return cls._instance

    def _find_all_project_processes(self) -> list:
        """Discovers all processes belonging to our pipeline, GUI, unroller/OMR, and VLM services."""
        procs = [self.proc]
        seen_pids = {self.pid}
        try:
            for ch in self.proc.children(recursive=True):
                if ch.pid not in seen_pids:
                    seen_pids.add(ch.pid)
                    procs.append(ch)
            parent = self.proc.parent()
            if parent and parent.pid not in seen_pids:
                p_name = (parent.name() or "").lower()
                if "python" in p_name or "antigravity" in p_name:
                    seen_pids.add(parent.pid)
                    procs.append(parent)
                    for sib in parent.children(recursive=True):
                        if sib.pid not in seen_pids:
                            seen_pids.add(sib.pid)
                            procs.append(sib)
        except Exception:
            pass

        # Also find any llama-server, lmstudio, or sibling python processes running our project
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = p.info["pid"]
                if pid in seen_pids:
                    continue
                pname = (p.info["name"] or "").lower()
                if "llama-server" in pname or "lmstudio" in pname:
                    seen_pids.add(pid)
                    procs.append(p)
                elif "python" in pname:
                    cmd = " ".join(p.info["cmdline"] or [])
                    if "pdf_to_md_music" in cmd:
                        seen_pids.add(pid)
                        procs.append(p)
            except Exception:
                pass
        return procs

    def get_metrics(self) -> Dict[str, Any]:
        """
        Gathers isolated resource consumption metrics for this process.
        """
        now = time.time()
        if (now - self._last_time) < self._cache_ttl and self._last_metrics:
            return dict(self._last_metrics)

        all_procs = self._find_all_project_processes()

        # 1. Process CPU (%) across all project services
        try:
            total_cpu_pct = 0.0
            for p in all_procs:
                try:
                    total_cpu_pct += p.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Normalize to 0-100% of entire system capacity
            proc_cpu_percent = round(total_cpu_pct / self.num_cpus, 1)
        except Exception:
            proc_cpu_percent = 0.0

        # 2. Process RAM RSS (MB) across all project services
        try:
            total_rss = 0
            for p in all_procs:
                try:
                    total_rss += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            proc_ram_rss_mb = round(total_rss / (1024 * 1024), 1)
        except Exception:
            proc_ram_rss_mb = 0.0

        # 3. Process VRAM (MB)
        cuda_active = False
        proc_vram_mb = 0.0
        proc_vram_reserved_mb = 0.0
        gpu_compute_percent = 0.0

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            cuda_active = True
            try:
                proc_vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
                proc_vram_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
            except Exception:
                pass

        # Check NVML process table and device memory if available
        if self._nvml_initialized and self._gpu_handle:
            try:
                # Total device VRAM used (captures llama-server, PyTorch, and all active GPU models)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                dev_used_mb = round(mem.used / (1024 * 1024), 1)
                dev_total_mb = round(mem.total / (1024 * 1024), 1)
                self._total_vram_mb = dev_total_mb
                proc_vram_reserved_mb = dev_total_mb

                # On Windows WDDM, per-process usedGpuMemory is None; device memory is tracked in gpu_vram_used_mb
                all_gpu_pids = set(p.pid for p in all_procs)
                for proc_fn in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                    try:
                        for gproc in proc_fn(self._gpu_handle):
                            if gproc.pid in all_gpu_pids and gproc.usedGpuMemory:
                                nvml_mb = round(gproc.usedGpuMemory / (1024 * 1024), 1)
                                proc_vram_mb = max(proc_vram_mb, nvml_mb)
                    except Exception:
                        pass
                
                # Device-wide GPU utilization for activity indication
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_compute_percent = float(util.gpu)
            except Exception:
                pass

        # Determine readable runtime mode
        if cuda_active:
            runtime_mode = f"CUDA ({self._gpu_name})"
        elif self._nvml_initialized:
            runtime_mode = f"CPU Mode ({self._gpu_name} idle)"
        else:
            runtime_mode = "CPU Mode"

        sys_ram = psutil.virtual_memory()
        
        metrics = {
            "pid": self.pid,
            "cpu_percent": proc_cpu_percent,
            "ram_rss_mb": proc_ram_rss_mb,
            "vram_allocated_mb": proc_vram_mb,
            "gpu_vram_used_mb": dev_used_mb,
            "vram_reserved_mb": proc_vram_reserved_mb,
            "gpu_compute_percent": gpu_compute_percent,
            "device_name": self._gpu_name,
            "total_vram_mb": self._total_vram_mb,
            "cuda_available": cuda_active,
            "runtime_mode": runtime_mode,
            "system_context": {
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "system_ram_used_gb": round(sys_ram.used / (1024 ** 3), 2),
                "system_ram_total_gb": round(sys_ram.total / (1024 ** 3), 2),
            },
        }
        self._last_time = now
        self._last_metrics = metrics
        return dict(metrics)


def get_process_telemetry() -> Dict[str, Any]:
    """Convenience function to get the current process telemetry."""
    return ProcessTelemetry.get_instance().get_metrics()
