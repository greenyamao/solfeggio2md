"""
Page Inspector & Multi-Mode Verification View.

Allows deep inspection of any processed book across 4 modes:
1. Scan vs Mask: BBox annotations vs masked page
2. Normalization: Raw crop vs deskewed staff with laser guidelines
3. Music Crop -> ABC / Humdrum code
4. Final Markdown Page
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    ComboBox,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    SegmentedWidget,
    SpinBox,
    SubtitleLabel,
    ToolButton,
)

from gui_qt.widgets.canvas_view import StaffGraphicsView


class InspectorView(QWidget):
    """Detailed page inspection and verification workspace."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.root_dir = Path(__file__).parent.parent.parent.resolve()
        self.output_root = self.root_dir / "output"

        self._books: List[str] = []
        self._current_page: int = 1
        self._max_page: int = 1

        self._init_ui()
        self.refresh_books()

    def _init_ui(self) -> None:
        self.setObjectName("InspectorView")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # 1. Header Toolbar
        main_layout.addWidget(self._create_toolbar())

        # 2. Split Workspace
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)

        # Left Canvas: Always displays original scan / crop
        left_box = QWidget(self.splitter)
        l_layout = QVBoxLayout(left_box)
        l_layout.setContentsMargins(0, 0, 4, 0)
        l_layout.setSpacing(6)

        self.lbl_left_title = CaptionLabel("ИСХОДНЫЙ СКАН / РАЗМЕТКА", left_box)
        self.lbl_left_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        l_layout.addWidget(self.lbl_left_title)

        self.canvas_left = StaffGraphicsView(left_box)
        l_layout.addWidget(self.canvas_left, stretch=1)

        self.splitter.addWidget(left_box)

        # Right Stack: Either Right Canvas or Code Editor
        right_box = QWidget(self.splitter)
        r_layout = QVBoxLayout(right_box)
        r_layout.setContentsMargins(4, 0, 0, 0)
        r_layout.setSpacing(6)

        r_header = QHBoxLayout()
        self.lbl_right_title = CaptionLabel("РЕЗУЛЬТАТ ОБРАБОТКИ", right_box)
        self.lbl_right_title.setStyleSheet("color: #4ade80; font-weight: 700; font-size: 11px;")
        r_header.addWidget(self.lbl_right_title)
        r_header.addStretch()

        self.btn_copy = PushButton(FluentIcon.COPY, "Копировать код", right_box)
        self.btn_copy.setVisible(False)
        self.btn_copy.clicked.connect(self._copy_code)
        r_header.addWidget(self.btn_copy)

        r_layout.addLayout(r_header)

        self.right_stack = QStackedWidget(right_box)

        # Page 0: Canvas for Modes 1 & 2
        self.canvas_right = StaffGraphicsView(self.right_stack)
        self.right_stack.addWidget(self.canvas_right)

        # Page 1: Text Editor for Modes 3 & 4
        self.text_editor = QPlainTextEdit(self.right_stack)
        self.text_editor.setReadOnly(True)
        self.text_editor.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b0f17;
                color: #e2e8f0;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 12px;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.right_stack.addWidget(self.text_editor)

        r_layout.addWidget(self.right_stack, stretch=1)

        self.splitter.addWidget(right_box)
        self.splitter.setSizes([500, 500])

        main_layout.addWidget(self.splitter, stretch=1)

    def _create_toolbar(self) -> CardWidget:
        bar = CardWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        # Book selector
        layout.addWidget(CaptionLabel("Книга:", bar))
        self.combo_books = ComboBox(bar)
        self.combo_books.setMinimumWidth(220)
        self.combo_books.currentIndexChanged.connect(self._on_book_changed)
        layout.addWidget(self.combo_books)

        # Page navigation
        layout.addWidget(CaptionLabel("Стр:", bar))
        self.btn_prev = ToolButton(FluentIcon.LEFT_ARROW, bar)
        self.btn_prev.clicked.connect(self._prev_page)
        layout.addWidget(self.btn_prev)

        self.spin_page = SpinBox(bar)
        self.spin_page.setRange(1, 1)
        self.spin_page.valueChanged.connect(self._on_page_changed)
        layout.addWidget(self.spin_page)

        self.lbl_max_page = CaptionLabel("/ 1", bar)
        layout.addWidget(self.lbl_max_page)

        self.btn_next = ToolButton(FluentIcon.RIGHT_ARROW, bar)
        self.btn_next.clicked.connect(self._next_page)
        layout.addWidget(self.btn_next)

        layout.addStretch()

        # Mode Selector
        self.mode_segmented = SegmentedWidget(bar)
        self.mode_segmented.addItem("mode1", "1. Скан vs Маска")
        self.mode_segmented.addItem("mode2", "2. Дескев")
        self.mode_segmented.addItem("mode3", "3. ABC Ноты")
        self.mode_segmented.addItem("mode4", "4. Markdown")
        self.mode_segmented.setCurrentItem("mode1")
        self.mode_segmented.currentItemChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_segmented)

        # Refresh
        self.btn_refresh = ToolButton(FluentIcon.SYNC, bar)
        self.btn_refresh.setToolTip("Обновить список книг")
        self.btn_refresh.clicked.connect(self.refresh_books)
        layout.addWidget(self.btn_refresh)

        return bar

    def refresh_books(self) -> None:
        self.combo_books.blockSignals(True)
        self.combo_books.clear()
        self._books.clear()

        if self.output_root.is_dir():
            for d in sorted(self.output_root.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    self._books.append(d.name)
                    self.combo_books.addItem(d.name)

        self.combo_books.blockSignals(False)
        if self._books:
            self._on_book_changed(0)

    def _on_book_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._books):
            return
        book_title = self._books[idx]
        book_dir = self.output_root / book_title
        raw_pages_dir = book_dir / "1_raw_pages"

        # Count pages
        pages = list(raw_pages_dir.glob("page_*_raw.png"))
        if not pages:
            pages = list(raw_pages_dir.glob("page_*.png"))

        self._max_page = max(1, len(pages))
        self.spin_page.blockSignals(True)
        self.spin_page.setMaximum(self._max_page)
        self.spin_page.setValue(1)
        self.spin_page.blockSignals(False)
        self.lbl_max_page.setText(f"/ {self._max_page}")

        self._current_page = 1
        self._load_current_view()

    def _prev_page(self) -> None:
        if self._current_page > 1:
            self.spin_page.setValue(self._current_page - 1)

    def _next_page(self) -> None:
        if self._current_page < self._max_page:
            self.spin_page.setValue(self._current_page + 1)

    def _on_page_changed(self, val: int) -> None:
        self._current_page = val
        self._load_current_view()

    def _on_mode_changed(self, key: str) -> None:
        self._load_current_view()

    def _load_current_view(self) -> None:
        if not self._books:
            return

        book_title = self.combo_books.currentText()
        book_dir = self.output_root / book_title
        p_num = self._current_page
        mode_key = self.mode_segmented.currentItem().property("name") if hasattr(self.mode_segmented.currentItem(), "property") else "mode1"
        # Fallback to text check if property is empty
        cur_text = self.mode_segmented.currentItem().text() if self.mode_segmented.currentItem() else "1. Скан vs Маска"

        raw_file = book_dir / "1_raw_pages" / f"page_{p_num:04d}_raw.png"
        if not raw_file.is_file():
            raw_file = book_dir / "1_raw_pages" / f"page_{p_num:04d}.png"

        deskew_file = book_dir / "1_raw_pages" / f"page_{p_num:04d}_deskew.png"
        mask_file = book_dir / "2_masked_pages" / f"page_{p_num:04d}_masked.png"
        debug_file = book_dir / "2_masked_pages" / f"page_{p_num:04d}_debug.png"
        md_file = book_dir / "4_final_pages" / f"page_{p_num:04d}.md"

        if "1. Скан" in cur_text:
            self.right_stack.setCurrentIndex(0)
            self.btn_copy.setVisible(False)
            self.lbl_left_title.setText(f"СТР. {p_num}: РАЗМЕТКА BBOX")
            self.lbl_right_title.setText(f"СТР. {p_num}: МАСКИРОВАННАЯ СТРАНИЦА")

            src_left = debug_file if debug_file.is_file() else raw_file
            if src_left.is_file():
                self.canvas_left.load_file(str(src_left))
            if mask_file.is_file():
                self.canvas_right.load_file(str(mask_file))

        elif "2. Дескев" in cur_text:
            self.right_stack.setCurrentIndex(0)
            self.btn_copy.setVisible(False)
            self.lbl_left_title.setText(f"СТР. {p_num}: ИСХОДНЫЙ СКАН")
            self.lbl_right_title.setText(f"СТР. {p_num}: 2D-DFT ВЫРАВНИВАНИЕ")

            if raw_file.is_file():
                self.canvas_left.load_file(str(raw_file))
            if deskew_file.is_file():
                self.canvas_right.load_file(str(deskew_file))

        elif "3. ABC" in cur_text:
            self.right_stack.setCurrentIndex(1)
            self.btn_copy.setVisible(True)
            self.lbl_left_title.setText(f"СТР. {p_num}: НОТНЫЕ СТАНЫ")
            self.lbl_right_title.setText(f"СТР. {p_num}: ДЕКОДИРОВАННЫЕ НОТЫ (ABC)")

            crops_dir = book_dir / "1_crops"
            crops = sorted(crops_dir.glob(f"*_P{p_num:04d}_*.png"))
            crops = [c for c in crops if not c.name.endswith("_deskew.png")]

            if crops:
                self.canvas_left.load_file(str(crops[0]))

            # Collect ABC codes
            abc_texts = []
            abcs_dir = book_dir / "3_abcs"
            for c in crops:
                abc_f = abcs_dir / f"{c.stem}.abc"
                if abc_f.is_file():
                    abc_texts.append(f"%% --- {c.name} ---\n" + abc_f.read_text(encoding="utf-8"))

            if abc_texts:
                self.text_editor.setPlainText("\n\n".join(abc_texts))
            else:
                self.text_editor.setPlainText("Ноты для данной страницы пока не распознаны.")

        elif "4. Markdown" in cur_text:
            self.right_stack.setCurrentIndex(1)
            self.btn_copy.setVisible(True)
            self.lbl_left_title.setText(f"СТР. {p_num}: СКАН СТРАНИЦЫ")
            self.lbl_right_title.setText(f"СТР. {p_num}: СОБРАННЫЙ MARKDOWN ДОКУМЕНТ")

            if raw_file.is_file():
                self.canvas_left.load_file(str(raw_file))

            if md_file.is_file():
                self.text_editor.setPlainText(md_file.read_text(encoding="utf-8"))
            else:
                self.text_editor.setPlainText(f"Финальный документ {md_file.name} ещё не собран.")

    def _copy_code(self) -> None:
        txt = self.text_editor.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            InfoBar.success(
                title="Скопировано",
                content="Код скопирован в буфер обмена.",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self,
            )
