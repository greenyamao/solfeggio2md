"""
Real-Time Telemetry HUD Ribbon Widget.

Displays high-contrast hardware and neural performance cards:
- CPU Load (% sum of all project services: GUI, worker, OMR, llama-server)
- RAM Usage (MB/GB sum of all project services)
- GPU Compute Load (% utilization from NVML)
- GPU VRAM Utilization (Used / Total GB hardware memory)
- VLM generation speed (pure tokens/sec)
- Page cycle latency (seconds)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import CaptionLabel, CardWidget


class MetricCard(CardWidget):
    """Compact metric card with prominent value and subtle caption."""

    def __init__(self, label: str, initial_val: str, unit: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.unit = unit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self.caption = CaptionLabel(label, self)
        self.caption.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 600; text-transform: uppercase;")
        layout.addWidget(self.caption)

        self.value_label = QLabel(initial_val, self)
        self.value_label.setStyleSheet("""
            QLabel {
                color: #38bdf8;
                font-size: 13px;
                font-weight: 700;
                font-family: 'Segoe UI Variable Display', 'Segoe UI', sans-serif;
            }
        """)
        layout.addWidget(self.value_label)

    def set_value(self, val: str) -> None:
        if self.unit:
            self.value_label.setText(f"{val} {self.unit}")
        else:
            self.value_label.setText(str(val))


class TelemetryHUDRibbon(QWidget):
    """Horizontal ribbon containing 6 live hardware and neural performance indicators."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. CPU (All Project Services)
        self.card_cpu = MetricCard("CPU (All Services)", "0.0 %", "", self)
        self.card_cpu.value_label.setStyleSheet("color: #38bdf8; font-size: 13px; font-weight: 700;")
        layout.addWidget(self.card_cpu)

        # 2. RAM (All Project Services)
        self.card_ram = MetricCard("RAM (Services)", "0 MB", "", self)
        self.card_ram.value_label.setStyleSheet("color: #60a5fa; font-size: 13px; font-weight: 700;")
        layout.addWidget(self.card_ram)

        # 3. GPU Compute Load
        self.card_gpu = MetricCard("GPU Compute", "0 %", "", self)
        self.card_gpu.value_label.setStyleSheet("color: #a78bfa; font-size: 13px; font-weight: 700;")
        layout.addWidget(self.card_gpu)

        # 4. GPU VRAM Usage (Used / Total)
        self.card_vram = MetricCard("VRAM Usage", "0.0 / 8.0 GB", "", self)
        self.card_vram.value_label.setStyleSheet("color: #c084fc; font-size: 13px; font-weight: 700;")
        layout.addWidget(self.card_vram)

        # 5. VLM Generation Speed
        self.card_speed = MetricCard("VLM Speed", "0.0 tok/s", "", self)
        self.card_speed.value_label.setStyleSheet("color: #4ade80; font-size: 13px; font-weight: 700;")
        layout.addWidget(self.card_speed)

        # 6. Page Cycle Latency
        self.card_time = MetricCard("Page Time", "0.0 s", "", self)
        self.card_time.value_label.setStyleSheet("color: #facc15; font-size: 13px; font-weight: 700;")
        layout.addWidget(self.card_time)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Updates all 6 live indicators across all project services and hardware."""
        # 1. CPU
        cpu = float(metrics.get("cpu_percent", 0.0))
        self.card_cpu.set_value(f"{cpu:.1f} %")
        sys_cpu = metrics.get("system_context", {}).get("system_cpu_percent", 0.0)
        self.card_cpu.setToolTip(f"Project Services: {cpu:.1f}% | Total System: {sys_cpu:.1f}%")

        # 2. RAM
        ram_mb = float(metrics.get("ram_rss_mb", 0.0))
        sys_ram = metrics.get("system_context", {})
        sys_used = float(sys_ram.get("system_ram_used_gb", 0.0))
        sys_tot = float(sys_ram.get("system_ram_total_gb", 0.0))
        if ram_mb >= 1024.0:
            ram_str = f"{ram_mb / 1024.0:.2f} GB"
        else:
            ram_str = f"{int(ram_mb)} MB"
        self.card_ram.set_value(ram_str)
        self.card_ram.setToolTip(f"Project Services: {ram_str} | Total System: {sys_used:.1f} / {sys_tot:.1f} GB")

        # 3. GPU Compute Load
        gpu_pct = float(metrics.get("gpu_compute_percent", 0.0))
        self.card_gpu.set_value(f"{int(gpu_pct)} %")
        dev_name = metrics.get("device_name", "GPU")
        self.card_gpu.setToolTip(f"Active Compute Device: {dev_name}")

        # 4. Hardware VRAM Usage (Used / Total)
        vram_used = float(metrics.get("gpu_vram_used_mb") or metrics.get("vram_allocated_mb") or 0.0)
        vram_total = float(metrics.get("total_vram_mb") or 8151.0)
        used_gb = vram_used / 1024.0
        total_gb = vram_total / 1024.0
        self.card_vram.set_value(f"{used_gb:.2f} / {total_gb:.1f} GB")
        if used_gb >= 7.2:
            self.card_vram.value_label.setStyleSheet("color: #fb923c; font-size: 13px; font-weight: 700;")
        else:
            self.card_vram.value_label.setStyleSheet("color: #c084fc; font-size: 13px; font-weight: 700;")
        self.card_vram.setToolTip(f"GPU Hardware VRAM: {vram_used:.0f} MB / {vram_total:.0f} MB")

        # 5. VLM Generation Speed
        spd = float(metrics.get("vlm_tok_per_sec", 0.0))
        self.card_speed.set_value(f"{spd:.1f} tok/s")

        # 6. Page Cycle Latency
        sec = float(metrics.get("last_page_seconds", 0.0))
        self.card_time.set_value(f"{sec:.1f} s")
