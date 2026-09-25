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
import re
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
        self._current_crops: List[Path] = []

        self._init_ui()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_books(preserve_current=True)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
            self._prev_page()
            event.accept()
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
            self._next_page()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _init_ui(self) -> None:
        self.setObjectName("InspectorView")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
                line-height: 1.4;
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
        layout.setSpacing(10)

        # Book selector
        layout.addWidget(CaptionLabel("Книга:", bar))
        self.combo_books = ComboBox(bar)
        self.combo_books.setMinimumWidth(240)
        self.combo_books.currentIndexChanged.connect(self._on_book_changed)
        layout.addWidget(self.combo_books)

        # Page navigation
        layout.addWidget(CaptionLabel("Стр:", bar))
        self.btn_prev = ToolButton(FluentIcon.LEFT_ARROW, bar)
        self.btn_prev.setToolTip("Предыдущая страница (Клавиша: Влево)")
        self.btn_prev.clicked.connect(self._prev_page)
        layout.addWidget(self.btn_prev)

        self.spin_page = SpinBox(bar)
        self.spin_page.setRange(1, 1)
        self.spin_page.valueChanged.connect(self._on_page_changed)
        layout.addWidget(self.spin_page)

        self.lbl_max_page = CaptionLabel("/ 1", bar)
        layout.addWidget(self.lbl_max_page)

        self.btn_next = ToolButton(FluentIcon.RIGHT_ARROW, bar)
        self.btn_next.setToolTip("Следующая страница (Клавиша: Вправо)")
        self.btn_next.clicked.connect(self._next_page)
        layout.addWidget(self.btn_next)

        # Staff Crop Selector (visible in Mode 2 & Mode 3 when multiple crops exist)
        self.lbl_crop = CaptionLabel("Стан:", bar)
        self.lbl_crop.setVisible(False)
        layout.addWidget(self.lbl_crop)

        self.combo_crops = ComboBox(bar)
        self.combo_crops.setMinimumWidth(130)
        self.combo_crops.setVisible(False)
        self.combo_crops.currentIndexChanged.connect(self._on_crop_changed)
        layout.addWidget(self.combo_crops)

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
        self.btn_refresh.setToolTip("Обновить книги и страницы")
        self.btn_refresh.clicked.connect(lambda: self.refresh_books(preserve_current=True))
        layout.addWidget(self.btn_refresh)

        return bar

    def _calculate_book_pages(self, book_dir: Path) -> int:
        """Dynamically scans book output directory and checkpoint for available pages."""
        max_p = 1

        # 1. Check checkpoint.json
        chk_file = book_dir / "checkpoint.json"
        if chk_file.is_file():
            try:
                data = json.loads(chk_file.read_text(encoding="utf-8"))
                slicing = data.get("phases", {}).get("slicing", {})
                tot = slicing.get("total_pages", 0)
                done = slicing.get("pages_done", 0)
                if tot > 0:
                    max_p = max(max_p, tot)
                elif done > 0:
                    max_p = max(max_p, done)
            except Exception:
                pass

        # 2. Check 2_masked_pages
        masked_dir = book_dir / "2_masked_pages"
        if masked_dir.is_dir():
            for f in masked_dir.glob("page_*_*.png"):
                parts = f.stem.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    max_p = max(max_p, int(parts[1]))

        # 3. Check 1_crops
        crops_dir = book_dir / "1_crops"
        if crops_dir.is_dir():
            for f in crops_dir.glob("*.png"):
                m = re.search(r"_P(\d+)_", f.stem)
                if m:
                    max_p = max(max_p, int(m.group(1)))

        # 4. Check 4_final_pages
        final_dir = book_dir / "4_final_pages"
        if final_dir.is_dir():
            for f in final_dir.glob("page_*.md"):
                parts = f.stem.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    max_p = max(max_p, int(parts[1]))

        return max(1, max_p)

    def _find_page_crops(self, crops_dir: Path, p_num: int) -> List[Path]:
        """Finds all non-deskew crop images belonging to page p_num."""
        if not crops_dir.is_dir():
            return []
        pat = re.compile(rf"_P0*{p_num}_S\d+", re.IGNORECASE)
        crops = []
        for f in crops_dir.glob("*.png"):
            if f.name.endswith("_deskew.png") or "preview" in f.name:
                continue
            if pat.search(f.name):
                crops.append(f)
        return sorted(crops)

    def refresh_books(self, preserve_current: bool = False) -> None:
        curr_book = self.combo_books.currentText() if preserve_current else None
        curr_page = self._current_page if preserve_current else 1

        self.combo_books.blockSignals(True)
        self.combo_books.clear()
        self._books.clear()

        if self.output_root.is_dir():
            for d in sorted(self.output_root.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    self._books.append(d.name)
                    self.combo_books.addItem(d.name)

        if not self._books:
            self.combo_books.blockSignals(False)
            self._current_page = 1
            self._max_page = 1
            self.spin_page.setRange(1, 1)
            self.spin_page.setValue(1)
            self.lbl_max_page.setText("/ 1")
            self.canvas_left.clear_view()
            self.canvas_right.clear_view()
            self.text_editor.clear()
            self._update_nav_buttons()
            return

        target_idx = 0
        if curr_book and curr_book in self._books:
            target_idx = self._books.index(curr_book)

        self.combo_books.setCurrentIndex(target_idx)
        self.combo_books.blockSignals(False)
        self._on_book_changed(target_idx, initial_page=curr_page)

    def _on_book_changed(self, idx: int, initial_page: int = 1) -> None:
        if idx < 0 or idx >= len(self._books):
            return
        book_title = self._books[idx]
        book_dir = self.output_root / book_title

        self._max_page = self._calculate_book_pages(book_dir)

        target_p = max(1, min(initial_page, self._max_page))
        self.spin_page.blockSignals(True)
        self.spin_page.setRange(1, self._max_page)
        self.spin_page.setValue(target_p)
        self.spin_page.blockSignals(False)

        self.lbl_max_page.setText(f"/ {self._max_page}")
        self._current_page = target_p
        self._update_nav_buttons()
        self._load_current_view()

    def _update_nav_buttons(self) -> None:
        self.btn_prev.setEnabled(self._current_page > 1)
        self.btn_next.setEnabled(self._current_page < self._max_page)

    def _prev_page(self) -> None:
        if self._current_page > 1:
            self.spin_page.setValue(self._current_page - 1)

    def _next_page(self) -> None:
        if self._current_page < self._max_page:
            self.spin_page.setValue(self._current_page + 1)

    def _on_page_changed(self, val: int) -> None:
        self._current_page = val
        self._update_nav_buttons()
        self._load_current_view()

    def _on_mode_changed(self, key: str) -> None:
        self._load_current_view()

    def _on_crop_changed(self, idx: int) -> None:
        if idx >= 0:
            self._load_current_view()

    def _load_current_view(self) -> None:
        if not self._books:
            return

        book_title = self.combo_books.currentText()
        book_dir = self.output_root / book_title
        p_num = self._current_page

        cur_text = self.mode_segmented.currentItem().text() if self.mode_segmented.currentItem() else "1. Скан vs Маска"

        # Standard file locations
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"

        debug_file = masked_dir / f"page_{p_num:04d}_debug.png"
        mask_file = masked_dir / f"page_{p_num:04d}_masked.png"
        raw_md_file = raw_md_dir / f"page_{p_num:04d}_raw.md"
        final_md_file = final_dir / f"page_{p_num:04d}.md"

        crops = self._find_page_crops(crops_dir, p_num)
        self._current_crops = crops

        # Update crops combo visibility
        is_crop_mode = ("2. Дескев" in cur_text or "3. ABC" in cur_text)
        if is_crop_mode and len(crops) > 0:
            self.lbl_crop.setVisible(True)
            self.combo_crops.setVisible(True)
            prev_crop_idx = self.combo_crops.currentIndex()
            self.combo_crops.blockSignals(True)
            self.combo_crops.clear()
            for i, c in enumerate(crops):
                tag_match = re.search(r"_S(\d+)_", c.name)
                tag_str = f"S{tag_match.group(1)}" if tag_match else f"S{i+1:02d}"
                self.combo_crops.addItem(f"Стан {i+1} ({tag_str})", c)
            if 0 <= prev_crop_idx < len(crops):
                self.combo_crops.setCurrentIndex(prev_crop_idx)
            else:
                self.combo_crops.setCurrentIndex(0)
            self.combo_crops.blockSignals(False)
        else:
            self.lbl_crop.setVisible(False)
            self.combo_crops.setVisible(False)

        selected_crop_idx = max(0, min(self.combo_crops.currentIndex(), len(crops) - 1)) if crops else 0
        active_crop = crops[selected_crop_idx] if crops else None

        # -------------------------------------------------------------
        # Mode 1: Scan vs Mask
        # -------------------------------------------------------------
        if "1. Скан" in cur_text:
            self.right_stack.setCurrentIndex(0)
            self.btn_copy.setVisible(False)
            self.lbl_left_title.setText(f"СТР. {p_num}: РАЗМЕТКА YOLO (BBOX)")
            self.lbl_right_title.setText(f"СТР. {p_num}: МАСКИРОВАННАЯ СТРАНИЦА")

            src_left = debug_file if debug_file.is_file() else mask_file
            if src_left and src_left.is_file():
                self.canvas_left.load_file(str(src_left))
            else:
                self.canvas_left.clear_view()
                self.lbl_left_title.setText(f"СТР. {p_num}: ЕЩЁ НЕ ОБРАБОТАНА (СЛАЙСИНГ)")

            if mask_file.is_file():
                self.canvas_right.load_file(str(mask_file))
            else:
                self.canvas_right.clear_view()
                self.lbl_right_title.setText(f"СТР. {p_num}: МАСКА НЕ СФОРМИРОВАНА")

        # -------------------------------------------------------------
        # Mode 2: Deskew
        # -------------------------------------------------------------
        elif "2. Дескев" in cur_text:
            self.right_stack.setCurrentIndex(0)
            self.btn_copy.setVisible(False)

            if active_crop is not None and active_crop.is_file():
                self.lbl_left_title.setText(f"СТР. {p_num} (СТАН {selected_crop_idx + 1}/{len(crops)}): ИСХОДНЫЙ КРОП")
                self.lbl_right_title.setText(f"СТР. {p_num} (СТАН {selected_crop_idx + 1}/{len(crops)}): 2D-DFT ВЫРАВНИВАНИЕ")

                self.canvas_left.load_file(str(active_crop))

                deskew_path = active_crop.parent / f"{active_crop.stem}_deskew.png"
                if deskew_path.is_file():
                    self.canvas_right.load_file(str(deskew_path))
                else:
                    self.canvas_right.load_file(str(active_crop))
            else:
                # No crops on this page: show page scan
                src_page = debug_file if debug_file.is_file() else mask_file
                if src_page and src_page.is_file():
                    self.canvas_left.load_file(str(src_page))
                else:
                    self.canvas_left.clear_view()
                self.canvas_right.clear_view()

                self.lbl_left_title.setText(f"СТР. {p_num}: ТЕКСТОВАЯ СТРАНИЦА")
                self.lbl_right_title.setText(f"СТР. {p_num}: НОТНЫХ СТАНОВ НЕ ОБНАРУЖЕНО")

        # -------------------------------------------------------------
        # Mode 3: ABC Notes
        # -------------------------------------------------------------
        elif "3. ABC" in cur_text:
            self.right_stack.setCurrentIndex(1)
            self.btn_copy.setVisible(True)

            if active_crop is not None and active_crop.is_file():
                self.lbl_left_title.setText(f"СТР. {p_num} (СТАН {selected_crop_idx + 1}/{len(crops)}): НОТНЫЙ СТАН")
                deskew_path = active_crop.parent / f"{active_crop.stem}_deskew.png"
                disp_crop = deskew_path if deskew_path.is_file() else active_crop
                self.canvas_left.load_file(str(disp_crop))
            else:
                src_page = debug_file if debug_file.is_file() else mask_file
                if src_page and src_page.is_file():
                    self.canvas_left.load_file(str(src_page))
                else:
                    self.canvas_left.clear_view()
                self.lbl_left_title.setText(f"СТР. {p_num}: НОТНЫХ СТАНОВ НЕ ОБНАРУЖЕНО")

            # Collect ABC codes
            abc_texts = []
            if crops:
                for i, c in enumerate(crops):
                    abc_f = c.with_suffix(".abc")
                    if abc_f.is_file() and abc_f.stat().st_size > 0:
                        code = abc_f.read_text(encoding="utf-8").strip()
                        abc_texts.append(f"% --- Стан {i + 1} ({c.name}) ---\n{code}")

            if abc_texts:
                self.text_editor.setPlainText("\n\n".join(abc_texts))
                self.lbl_right_title.setText(f"СТР. {p_num}: ДЕКОДИРОВАННЫЕ НОТЫ (ABC)")
            elif crops:
                self.text_editor.setPlainText(
                    f"На странице {p_num} обнаружено {len(crops)} нотных станов.\n"
                    "Они находятся в очереди OMR-распознавания модели Transcoda-59M."
                )
                self.lbl_right_title.setText(f"СТР. {p_num}: ОЖИДАЕТ OMR РАСПОЗНАВАНИЯ")
            else:
                self.text_editor.setPlainText(f"На странице {p_num} нотные станы не обнаружены (текстовая страница).")
                self.lbl_right_title.setText(f"СТР. {p_num}: НОТ НЕТ")

        # -------------------------------------------------------------
        # Mode 4: Markdown
        # -------------------------------------------------------------
        elif "4. Markdown" in cur_text:
            self.right_stack.setCurrentIndex(1)
            self.btn_copy.setVisible(True)
            self.lbl_left_title.setText(f"СТР. {p_num}: СКАН СТРАНИЦЫ")

            src_page = debug_file if debug_file.is_file() else mask_file
            if src_page and src_page.is_file():
                self.canvas_left.load_file(str(src_page))
            else:
                self.canvas_left.clear_view()

            if final_md_file.is_file():
                self.text_editor.setPlainText(final_md_file.read_text(encoding="utf-8"))
                self.lbl_right_title.setText(f"СТР. {p_num}: СОБРАННЫЙ MARKDOWN ДОКУМЕНТ")
            elif raw_md_file.is_file():
                self.text_editor.setPlainText(raw_md_file.read_text(encoding="utf-8"))
                self.lbl_right_title.setText(f"СТР. {p_num}: ЧЕРНОВИК VLM OCR (БЕЗ НОТ)")
            else:
                self.text_editor.setPlainText(
                    f"Markdown документ для страницы {p_num} ещё не сформирован.\n"
                    "Страница ожидает этапа VLM OCR (Qwen 3.5) или финальной сборки."
                )
                self.lbl_right_title.setText(f"СТР. {p_num}: ОЖИДАЕТ ОБРАБОТКИ")

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
