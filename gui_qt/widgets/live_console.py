"""
High-Contrast Monospace Real-Time Neural & Pipeline Console Widget.

Displays structured log output with color-coded tags ([YOLO], [OMR], [VLM], [ASSEMBLY]),
interactive auto-scroll toggle, and copyable text buffer.
"""

from __future__ import annotations

import html
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import CaptionLabel, FluentIcon, SubtitleLabel, SwitchButton, ToolButton


class LiveConsoleWidget(QWidget):
    """Monospace interactive live terminal log viewer."""

    TAG_COLORS = {
        "YOLO": "#38bdf8",      # Cyan / Light Blue
        "OMR": "#c084fc",       # Purple
        "VLM": "#4ade80",       # Neon Green
        "ASSEMBLY": "#facc15",  # Gold
        "SYSTEM": "#94a3b8",    # Slate Grey
        "WARN": "#fb923c",      # Orange
        "ERROR": "#f87171",     # Coral Red
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._auto_scroll: bool = True
        self._line_count: int = 0

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header bar
        header = QWidget(self)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(6, 4, 6, 4)
        h_layout.setSpacing(10)

        self.title_label = CaptionLabel("LIVE PIPELINE & NEURAL LOG (0 lines)", header)
        self.title_label.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 11px;")
        h_layout.addWidget(self.title_label)

        h_layout.addStretch()

        # Auto-scroll switch
        self.scroll_switch = SwitchButton("Autoscroll", header)
        self.scroll_switch.setChecked(True)
        self.scroll_switch.checkedChanged.connect(self._on_auto_scroll_changed)
        h_layout.addWidget(self.scroll_switch)

        # Clear button
        self.clear_btn = ToolButton(FluentIcon.DELETE, header)
        self.clear_btn.setToolTip("Clear log")
        self.clear_btn.clicked.connect(self.clear_logs)
        h_layout.addWidget(self.clear_btn)

        layout.addWidget(header)

        # Text area
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumBlockCount(1000)

        font = QFont("Cascadia Code", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_edit.setFont(font)

        self.text_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #080c14;
                color: #e2e8f0;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #334155;
            }
        """)

        layout.addWidget(self.text_edit, stretch=1)

    def _on_auto_scroll_changed(self, checked: bool) -> None:
        self._auto_scroll = checked

    def append_log(self, entry: Dict[str, Any]) -> None:
        """Appends a single structured log line with syntax coloring."""
        time_str = html.escape(entry.get("time", ""))
        tag = html.escape(entry.get("tag", "SYSTEM"))
        msg = html.escape(entry.get("msg", ""))
        level = entry.get("level", "INFO")

        color = self.TAG_COLORS.get(tag, "#94a3b8")
        if level == "ERROR":
            color = self.TAG_COLORS["ERROR"]
        elif level == "WARN":
            color = self.TAG_COLORS["WARN"]

        line_html = (
            f"<span style='color: #64748b;'>{time_str}</span> "
            f"<b style='color: {color};'>[{tag}]</b> "
            f"<span style='color: #f1f5f9;'>{msg}</span>"
        )

        self.text_edit.appendHtml(line_html)
        self._line_count += 1
        self.title_label.setText(f"LIVE PIPELINE & NEURAL LOG ({self._line_count} lines)")

        if self._auto_scroll:
            self.text_edit.moveCursor(QTextCursor.MoveOperation.End)

    def clear_logs(self) -> None:
        """Clears the console log window."""
        self.text_edit.clear()
        self._line_count = 0
        self.title_label.setText("LIVE PIPELINE & NEURAL LOG (0 lines)")
