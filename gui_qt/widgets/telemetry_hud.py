"""
Real-Time Telemetry HUD Ribbon Widget.

Displays high-contrast hardware and neural performance cards:
- VLM generation speed (tokens/sec)
- Last page end-to-end processing time
- GPU VRAM utilization
- Output tokens count
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
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
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self.caption = CaptionLabel(label, self)
        self.caption.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600; text-transform: uppercase;")
        layout.addWidget(self.caption)

        self.value_label = QLabel(initial_val, self)
        self.value_label.setStyleSheet("""
            QLabel {
                color: #38bdf8;
                font-size: 16px;
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
    """Horizontal ribbon containing 4 live telemetry indicators."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.card_speed = MetricCard("VLM Скорость", "0.0", "tok/s", self)
        self.card_speed.value_label.setStyleSheet("color: #4ade80; font-size: 16px; font-weight: 700;")
        layout.addWidget(self.card_speed)

        self.card_time = MetricCard("Время стр.", "0.0", "с", self)
        self.card_time.value_label.setStyleSheet("color: #38bdf8; font-size: 16px; font-weight: 700;")
        layout.addWidget(self.card_time)

        self.card_vram = MetricCard("VRAM Память", "0.0", "MB", self)
        self.card_vram.value_label.setStyleSheet("color: #c084fc; font-size: 16px; font-weight: 700;")
        layout.addWidget(self.card_vram)

        self.card_tokens = MetricCard("Токены вывода", "0", "tok", self)
        self.card_tokens.value_label.setStyleSheet("color: #facc15; font-size: 16px; font-weight: 700;")
        layout.addWidget(self.card_tokens)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Updates all 4 metric values from pipeline metrics dictionary."""
        # 1. VLM Speed
        spd = float(metrics.get("vlm_tok_per_sec", 0.0))
        self.card_speed.set_value(f"{spd:.1f}")

        # 2. Last page latency
        sec = float(metrics.get("last_page_seconds", 0.0))
        self.card_time.set_value(f"{sec:.1f}")

        # 3. VRAM
        vram = float(metrics.get("vram_allocated_mb", 0.0))
        if vram >= 1024:
            self.card_vram.unit = "GB"
            self.card_vram.set_value(f"{vram / 1024.0:.2f}")
        else:
            self.card_vram.unit = "MB"
            self.card_vram.set_value(f"{int(vram)}")

        # 4. Output tokens
        toks = int(metrics.get("last_page_tokens", 0))
        self.card_tokens.set_value(str(toks))
