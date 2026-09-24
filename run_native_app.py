"""
Windows 11 Native High-Performance Workbench Runner.

Launches the PySide6 + QFluentWidgets graphical desktop application
with hardware-accelerated QGraphicsView rendering and zero-copy shared memory IPC.

Usage:
    .\\.venv\\Scripts\\python.exe run_native_app.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.windows_perf import enable_windows_high_performance
from gui_qt.app_window import MainWindow


def main() -> None:
    # Mandatory for Windows multiprocessing spawn
    mp.freeze_support()

    # Hardware power optimization and timer resolution
    enable_windows_high_performance()

    # High DPI display scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PDF to Markdown Music Workbench")
    app.setOrganizationName("Antigravity")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
