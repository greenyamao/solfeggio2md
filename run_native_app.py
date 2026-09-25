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

    # Suppress non-critical third-party deprecation notices (torch pytree, triton flop_counter, transformers)
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Hardware power optimization and timer resolution
    enable_windows_high_performance()

    # High DPI display scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PDF to Markdown Music Workbench")
    app.setOrganizationName("Antigravity")

    # Ensure font has a positive pointSize on Windows to avoid QFont::setPointSize warnings
    default_font = app.font()
    if default_font.pointSize() <= 0:
        default_font.setPointSize(9)
        app.setFont(default_font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
