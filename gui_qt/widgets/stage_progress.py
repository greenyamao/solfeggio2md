"""
4-Phase Pipeline Stepper and Progress Tracker Widget.

Visualizes progress across the 4 batch passes:
1. Slicing, Dewarping & YOLO Layout Detection
2. Mini-Batch Neural OMR (Transcoda-59M)
3. Vision-Language Model OCR (Qwen3.5-9B)
4. Markdown Music Assembly
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
from qfluentwidgets import CaptionLabel, CardWidget, ProgressBar, SubtitleLabel


class PhaseRow(QWidget):
    """A single phase indicator row with title, progress bar, and status pill."""

    def __init__(self, step_num: int, title: str, subtitle: str, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header row
        h_row = QWidget(self)
        hl = QHBoxLayout(h_row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)

        # Step badge
        self.num_badge = QLabel(f"PHASE {step_num}", h_row)
        self.num_badge.setStyleSheet("""
            QLabel {
                background-color: #1e293b;
                color: #38bdf8;
                font-weight: 700;
                font-size: 10px;
                padding: 2px 6px;
                border-radius: 4px;
            }
        """)
        hl.addWidget(self.num_badge)

        # Title
        self.title_label = QLabel(title, h_row)
        self.title_label.setStyleSheet("color: #f8fafc; font-weight: 600; font-size: 12px;")
        hl.addWidget(self.title_label)

        hl.addStretch()

        # Status badge
        self.status_pill = QLabel("Queued", h_row)
        self.status_pill.setStyleSheet("""
            QLabel {
                background-color: #0f172a;
                color: #64748b;
                font-weight: 600;
                font-size: 10px;
                padding: 2px 8px;
                border: 1px solid #1e293b;
                border-radius: 10px;
            }
        """)
        hl.addWidget(self.status_pill)

        layout.addWidget(h_row)

        # Progress bar
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        # Detail text
        self.detail_label = CaptionLabel(subtitle, self)
        self.detail_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.detail_label)

    def set_progress(self, pct: float, done: int, total: int, status: str, detail: str = "") -> None:
        self.progress_bar.setValue(int(pct))

        if detail:
            self.detail_label.setText(detail)
        elif total > 0:
            self.detail_label.setText(f"{done} of {total} ({pct:.1f}%)")

        if status == "running":
            self.status_pill.setText("Running...")
            self.status_pill.setStyleSheet("""
                QLabel {
                    background-color: #0284c7;
                    color: #ffffff;
                    font-weight: 700;
                    font-size: 10px;
                    padding: 2px 8px;
                    border-radius: 10px;
                }
            """)
        elif status == "finished" or pct >= 100.0:
            self.status_pill.setText("Done")
            self.status_pill.setStyleSheet("""
                QLabel {
                    background-color: #166534;
                    color: #4ade80;
                    font-weight: 700;
                    font-size: 10px;
                    padding: 2px 8px;
                    border-radius: 10px;
                }
            """)
        elif status == "skipped":
            self.status_pill.setText("Skipped")
            self.status_pill.setStyleSheet("""
                QLabel {
                    background-color: #334155;
                    color: #94a3b8;
                    font-weight: 600;
                    font-size: 10px;
                    padding: 2px 8px;
                    border-radius: 10px;
                }
            """)
        else:
            self.status_pill.setText("Queued")
            self.status_pill.setStyleSheet("""
                QLabel {
                    background-color: #0f172a;
                    color: #64748b;
                    font-weight: 600;
                    font-size: 10px;
                    padding: 2px 8px;
                    border: 1px solid #1e293b;
                    border-radius: 10px;
                }
            """)


class StageProgressWidget(CardWidget):
    """Card containing the 3 pipeline stage rows."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        header = CaptionLabel("PIPELINE PROCESSING STAGES", self)
        header.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase;")
        layout.addWidget(header)

        self.row1 = PhaseRow(1, "Slicing & YOLO OLA", "Spread split, 2D-DFT deskew, staff detection", self)
        layout.addWidget(self.row1)

        self.row2 = PhaseRow(2, "Music Recognition (OMR)", "Mini-batched Transcoda-59M to ABC & Humdrum", self)
        layout.addWidget(self.row2)

        self.row3 = PhaseRow(3, "Text & Music Integration", "Qwen VLM text transcription with embedded ABC notes", self)
        layout.addWidget(self.row3)

    def update_phases(self, phase_data: Dict[str, Any]) -> None:
        """Updates all 3 rows from the pipeline phase_progress metrics."""
        p1 = phase_data.get("phase1", {})
        self.row1.set_progress(
            p1.get("pct", 0.0), p1.get("done", 0), p1.get("total", 0),
            p1.get("status", "pending"), p1.get("detail", "")
        )

        p2 = phase_data.get("phase2", {})
        self.row2.set_progress(
            p2.get("pct", 0.0), p2.get("done", 0), p2.get("total", 0),
            p2.get("status", "pending"), p2.get("detail", "")
        )

        p3 = phase_data.get("phase3", {})
        self.row3.set_progress(
            p3.get("pct", 0.0), p3.get("done", 0), p3.get("total", 0),
            p3.get("status", "pending"), p3.get("detail", "")
        )
