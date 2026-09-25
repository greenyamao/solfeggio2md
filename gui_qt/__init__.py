"""
High-Performance Windows 11 Native GUI Package for pdf_to_md_music.
Built with PySide6, QFluentWidgets, QGraphicsView, and Zero-Copy Shared Memory IPC.
"""

import io
import sys

# Silence third-party library promotional banner on import
_orig_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    import qfluentwidgets
finally:
    sys.stdout = _orig_stdout

from gui_qt.app_window import MainWindow

__all__ = ["MainWindow"]

