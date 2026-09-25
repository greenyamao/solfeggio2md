"""
Main Windows 11 Fluent Window Application Shell.

Integrates:
- Modern Windows 11 dark theme with mica/acrylic effects
- Zero-scroll viewport layout
- Sidebar navigation between Pipeline, Inspector, and Settings
- Isolated ML Pipeline Worker Process lifecycle
- Zero-Copy Shared Memory Frame Channel cleanup
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from qfluentwidgets import (
    FluentIcon,
    FluentWindow,
    NavigationItemPosition,
    Theme,
    setTheme,
)

from gui_qt.ipc.shared_frame import SharedFrameReader
from gui_qt.ipc.worker_process import MLPipelineProcessClient
from gui_qt.views.inspector_view import InspectorView
from gui_qt.views.pipeline_view import PipelineView
from gui_qt.views.settings_view import SettingsView


class MainWindow(FluentWindow):
    """Main desktop application window."""

    def __init__(self):
        super().__init__()

        # Force dark mode matching high-contrast engineering dashboard
        setTheme(Theme.DARK)

        # Initialize Zero-Copy Shared Memory Frame Reader
        self.frame_reader = SharedFrameReader()
        self.frame_reader.create()

        # Initialize Isolated ML Pipeline Worker Process Client
        self.worker_client = MLPipelineProcessClient()
        self.worker_client.start()

        # Build UI Views
        self._init_views()
        self._init_window()

    def _init_views(self) -> None:
        # 1. Pipeline View (Main Dashboard)
        self.pipeline_view = PipelineView(self.worker_client, self.frame_reader, self)
        self.addSubInterface(
            self.pipeline_view,
            FluentIcon.SPEED_HIGH,
            "Pipeline",
        )

        # 2. Page Inspector View
        self.inspector_view = InspectorView(self)
        self.addSubInterface(
            self.inspector_view,
            FluentIcon.VIEW,
            "Inspector",
        )

        # 3. Settings View (Bottom Position)
        self.settings_view = SettingsView(self.worker_client, self)
        self.addSubInterface(
            self.settings_view,
            FluentIcon.SETTING,
            "Settings",
            NavigationItemPosition.BOTTOM,
        )

    def _init_window(self) -> None:
        self.resize(1340, 840)
        self.setMinimumSize(1100, 720)
        self.setWindowTitle("PDF to Markdown Music — Windows 11 Native Workbench")
        # Disable sliding transition animation to prevent stutter and visual tearing
        if hasattr(self, "stackedWidget"):
            self.stackedWidget.setAnimationEnabled(False)


    def closeEvent(self, event) -> None:
        """Gracefully terminates worker child process and unlinks shared memory."""
        try:
            if hasattr(self, "worker_client"):
                self.worker_client.shutdown(timeout=1.5)
        except Exception:
            pass

        try:
            if hasattr(self, "frame_reader"):
                self.frame_reader.close_and_cleanup()
        except Exception:
            pass

        super().closeEvent(event)
