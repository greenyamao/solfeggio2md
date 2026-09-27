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
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
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
        self._bridge = None
        self._validator = None

        self._init_ui()

    @property
    def bridge(self):
        if self._bridge is None:
            from core.abc_bridge import ABCBridge
            self._bridge = ABCBridge()
        return self._bridge

    @property
    def validator(self):
        if self._validator is None:
            from core.notation_validator import NotationValidator
            self._validator = NotationValidator()
        return self._validator

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_books(preserve_current=True)
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_PageUp, Qt.Key.Key_A):
            self._prev_page()
            event.accept()
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_PageDown, Qt.Key.Key_D):
            self._next_page()
            event.accept()
        elif key == Qt.Key.Key_Home:
            self._set_page(1)
            event.accept()
        elif key == Qt.Key.Key_End:
            self._set_page(self._max_page)
            event.accept()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, watched, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            is_spinbox_input = (
                watched == self.spin_page
                or (hasattr(self.spin_page, "lineEdit") and watched == self.spin_page.lineEdit())
            )
            if key in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
                self._prev_page()
                return True
            elif key in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
                self._next_page()
                return True
            elif key == Qt.Key.Key_Home and not is_spinbox_input:
                self._set_page(1)
                return True
            elif key == Qt.Key.Key_End and not is_spinbox_input:
                self._set_page(self._max_page)
                return True
            elif key == Qt.Key.Key_A and not is_spinbox_input:
                self._prev_page()
                return True
            elif key == Qt.Key.Key_D and not is_spinbox_input:
                self._next_page()
                return True

        elif event.type() == QEvent.Type.Wheel:
            # Allow mouse wheel to flip pages when scrolling over navigation widgets
            if watched in (
                self.toolbar_card,
                self.spin_page,
                self.lbl_max_page,
                self.btn_prev,
                self.btn_next,
            ):
                angle = event.angleDelta().y()
                if angle > 0:
                    self._prev_page()
                    return True
                elif angle < 0:
                    self._next_page()
                    return True

        return super().eventFilter(watched, event)

    def _init_ui(self) -> None:
        self.setObjectName("InspectorView")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # 1. Header Toolbar
        self.toolbar_card = self._create_toolbar()
        main_layout.addWidget(self.toolbar_card)

        # 2. Split Workspace
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)

        # Left Canvas: Always displays original scan / crop
        left_box = QWidget(self.splitter)
        l_layout = QVBoxLayout(left_box)
        l_layout.setContentsMargins(0, 0, 4, 0)
        l_layout.setSpacing(6)

        self.lbl_left_title = CaptionLabel("SOURCE SCAN / ANNOTATION", left_box)
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
        self.lbl_right_title = CaptionLabel("PROCESSING RESULT", right_box)
        self.lbl_right_title.setStyleSheet("color: #4ade80; font-weight: 700; font-size: 11px;")
        r_header.addWidget(self.lbl_right_title)
        r_header.addStretch()

        self.btn_copy = PushButton(FluentIcon.COPY, "Copy Code", right_box)
        self.btn_copy.setVisible(False)
        self.btn_copy.clicked.connect(self._copy_code)
        r_header.addWidget(self.btn_copy)

        r_layout.addLayout(r_header)

        self.right_stack = QStackedWidget(right_box)

        # Page 0: Canvas for Modes 1 & 2
        self.canvas_right = StaffGraphicsView(self.right_stack)
        self.right_stack.addWidget(self.canvas_right)

        # Page 1: Mode 3: Integrated Quality Status Banner + Verovio SVG & Code Tabs
        self.mode3_container = QWidget(self.right_stack)
        m3_layout = QVBoxLayout(self.mode3_container)
        m3_layout.setContentsMargins(0, 0, 0, 0)
        m3_layout.setSpacing(6)

        # Quality / Invariant Diagnostic Banner
        self.card_val = CardWidget(self.mode3_container)
        card_val_layout = QVBoxLayout(self.card_val)
        card_val_layout.setContentsMargins(12, 8, 12, 8)
        card_val_layout.setSpacing(2)

        self.lbl_val_status = SubtitleLabel("Score: 100% (PASS)", self.card_val)
        self.lbl_val_status.setStyleSheet("color: #4ade80; font-weight: 700; font-size: 13px;")
        card_val_layout.addWidget(self.lbl_val_status)

        self.lbl_val_details = CaptionLabel("No structural anomalies detected", self.card_val)
        self.lbl_val_details.setWordWrap(True)
        card_val_layout.addWidget(self.lbl_val_details)

        m3_layout.addWidget(self.card_val)

        # Tabs: Vector Score vs Code
        self.tabs_mode3 = QTabWidget(self.mode3_container)
        self.tabs_mode3.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1e293b;
                border-radius: 6px;
                background-color: #0b0f17;
            }
            QTabBar::tab {
                background: #161b22;
                color: #8b9bb4;
                padding: 6px 14px;
                border: 1px solid #21262d;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-size: 11px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #1f2937;
                color: #ffffff;
                border-color: #388bfd;
            }
        """)

        # Tab 0: Vector Score Canvas (Aspect-ratio preserved, interactive zoom & pan)
        self.canvas_score = StaffGraphicsView(self.tabs_mode3)
        self.tabs_mode3.addTab(self.canvas_score, "Verovio Vector Score")

        # Tab 1: Text Editor for ABC / Kern
        self.text_editor = QPlainTextEdit(self.tabs_mode3)
        self.text_editor.setReadOnly(True)
        self.text_editor.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b0f17;
                color: #e2e8f0;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 12px;
                border: none;
                padding: 10px;
                line-height: 1.4;
            }
        """)
        self.tabs_mode3.addTab(self.text_editor, "Raw ABC / Humdrum Kern")

        m3_layout.addWidget(self.tabs_mode3, stretch=1)
        self.right_stack.addWidget(self.mode3_container)

        # Page 2: Text Editor for Mode 4 (Markdown)
        self.text_editor_md = QPlainTextEdit(self.right_stack)
        self.text_editor_md.setReadOnly(True)
        self.text_editor_md.setStyleSheet("""
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
        self.right_stack.addWidget(self.text_editor_md)

        r_layout.addWidget(self.right_stack, stretch=1)

        self.splitter.addWidget(right_box)
        self.splitter.setSizes([500, 500])

        main_layout.addWidget(self.splitter, stretch=1)

        # Install universal arrow key and scroll event filters across all child components
        filter_targets = [
            self,
            self.toolbar_card,
            self.canvas_left,
            self.canvas_right,
            self.canvas_score,
            self.text_editor,
            self.splitter,
            self.btn_prev,
            self.btn_next,
            self.btn_refresh,
            self.combo_books,
            self.combo_crops,
            self.mode_segmented,
            self.spin_page,
        ]
        for w in filter_targets:
            w.installEventFilter(self)
            if hasattr(w, "viewport") and w.viewport():
                w.viewport().installEventFilter(self)

    def _create_toolbar(self) -> CardWidget:
        bar = CardWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Book selector
        layout.addWidget(CaptionLabel("Book:", bar))
        self.combo_books = ComboBox(bar)
        self.combo_books.setMinimumWidth(240)
        self.combo_books.currentIndexChanged.connect(self._on_book_changed)
        layout.addWidget(self.combo_books)

        # Page navigation
        layout.addWidget(CaptionLabel("Page:", bar))
        self.btn_prev = ToolButton(FluentIcon.LEFT_ARROW, bar)
        self.btn_prev.setToolTip("Previous page (Keys: Left, A, PageUp)")
        self.btn_prev.clicked.connect(self._prev_page)
        layout.addWidget(self.btn_prev)

        self.spin_page = SpinBox(bar)
        self.spin_page.setRange(1, 1)
        self.spin_page.valueChanged.connect(self._on_page_changed)
        layout.addWidget(self.spin_page)

        self.lbl_max_page = CaptionLabel("/ 1", bar)
        layout.addWidget(self.lbl_max_page)

        self.btn_next = ToolButton(FluentIcon.RIGHT_ARROW, bar)
        self.btn_next.setToolTip("Next page (Keys: Right, D, PageDown)")
        self.btn_next.clicked.connect(self._next_page)
        layout.addWidget(self.btn_next)

        # Staff Crop Selector (visible in Mode 2 & Mode 3 when multiple crops exist)
        self.lbl_crop = CaptionLabel("Staff:", bar)
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
        self.mode_segmented.addItem("mode1", "1. Scan vs Mask")
        self.mode_segmented.addItem("mode2", "2. Deskew")
        self.mode_segmented.addItem("mode3", "3. ABC Notes")
        self.mode_segmented.addItem("mode4", "4. Markdown")
        self.mode_segmented.setCurrentItem("mode1")
        self.mode_segmented.currentItemChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_segmented)

        # Refresh
        self.btn_refresh = ToolButton(FluentIcon.SYNC, bar)
        self.btn_refresh.setToolTip("Refresh books and pages")
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
                m = re.search(r"(?:^|_)P(\d+)_", f.stem)
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
        pat = re.compile(rf"(?:^|_)P0*{p_num}_S\d+", re.IGNORECASE)
        crops = []
        for f in sorted(crops_dir.glob("*.png")):
            if f.name.endswith("_deskew.png") or "preview" in f.name:
                continue
            if pat.search(f.name):
                crops.append(f)
        return crops

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

    def _set_page(self, new_page: int) -> None:
        """Single source of truth for page changes."""
        new_page = max(1, min(new_page, self._max_page))
        if new_page == self._current_page and self.spin_page.value() == new_page:
            return
        self._current_page = new_page
        self.spin_page.blockSignals(True)
        self.spin_page.setValue(new_page)
        self.spin_page.blockSignals(False)
        self._update_nav_buttons()
        self._load_current_view()

    def _prev_page(self) -> None:
        if self._current_page > 1:
            self._set_page(self._current_page - 1)

    def _next_page(self) -> None:
        if self._current_page < self._max_page:
            self._set_page(self._current_page + 1)

    def _on_page_changed(self, val: int) -> None:
        self._set_page(val)

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

        cur_text = self.mode_segmented.currentItem().text() if self.mode_segmented.currentItem() else "1. Scan vs Mask"

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
        is_crop_mode = ("2. Deskew" in cur_text or "3. ABC" in cur_text)
        if is_crop_mode and len(crops) > 0:
            self.lbl_crop.setVisible(True)
            self.combo_crops.setVisible(True)
            prev_crop_idx = self.combo_crops.currentIndex()
            self.combo_crops.blockSignals(True)
            self.combo_crops.clear()
            for i, c in enumerate(crops):
                tag_match = re.search(r"_S(\d+)(?:_|\.)", c.name)
                tag_str = f"S{tag_match.group(1)}" if tag_match else f"S{i+1:02d}"
                self.combo_crops.addItem(f"Staff {i+1} ({tag_str})", c)
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
        if "1. Scan" in cur_text:
            self.right_stack.setCurrentIndex(0)
            self.btn_copy.setVisible(False)
            self.lbl_left_title.setText(f"PAGE {p_num}: YOLO BBOX ANNOTATION")
            self.lbl_right_title.setText(f"PAGE {p_num}: MASKED PAGE")

            src_left = debug_file if debug_file.is_file() else mask_file
            if src_left and src_left.is_file():
                self.canvas_left.load_file(str(src_left))
            else:
                self.canvas_left.clear_view()
                self.lbl_left_title.setText(f"PAGE {p_num}: NOT YET PROCESSED (SLICING)")

            if mask_file.is_file():
                self.canvas_right.load_file(str(mask_file))
            else:
                self.canvas_right.clear_view()
                self.lbl_right_title.setText(f"PAGE {p_num}: MASK NOT GENERATED")

        # -------------------------------------------------------------
        # Mode 2: Deskew
        # -------------------------------------------------------------
        elif "2. Deskew" in cur_text:
            self.right_stack.setCurrentIndex(0)
            self.btn_copy.setVisible(False)

            if active_crop is not None and active_crop.is_file():
                self.lbl_left_title.setText(f"PAGE {p_num} (STAFF {selected_crop_idx + 1}/{len(crops)}): RAW CROP")
                self.lbl_right_title.setText(f"PAGE {p_num} (STAFF {selected_crop_idx + 1}/{len(crops)}): 2D-DFT DESKEW")

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

                self.lbl_left_title.setText(f"PAGE {p_num}: TEXT PAGE")
                self.lbl_right_title.setText(f"PAGE {p_num}: NO STAVES DETECTED")

        # -------------------------------------------------------------
        # Mode 3: ABC Notes
        # -------------------------------------------------------------
        elif "3. ABC" in cur_text:
            self.right_stack.setCurrentIndex(1)
            self.btn_copy.setVisible(True)

            if active_crop is not None and active_crop.is_file():
                self.lbl_left_title.setText(f"PAGE {p_num} (STAFF {selected_crop_idx + 1}/{len(crops)}): MUSIC STAFF")
                deskew_path = active_crop.parent / f"{active_crop.stem}_deskew.png"
                disp_crop = deskew_path if deskew_path.is_file() else active_crop
                self.canvas_left.load_file(str(disp_crop))

                # Check .kern and .abc for active crop
                kern_file = active_crop.with_suffix(".kern")
                abc_file = active_crop.with_suffix(".abc")
                raw_kern = kern_file.read_text(encoding="utf-8").strip() if kern_file.is_file() else ""
                abc_text = abc_file.read_text(encoding="utf-8").strip() if abc_file.is_file() else ""

                if raw_kern or abc_text:
                    # Run automated validation under the hood
                    rep = self.validator.validate(raw_kern=raw_kern, abc_text=abc_text)
                    score_val = int(round(rep.score * 100))

                    if rep.is_valid and not rep.anomalies:
                        self.lbl_val_status.setText(f"Staff {selected_crop_idx + 1} — Score: {score_val}% (VERIFIED CLEAN)")
                        self.lbl_val_status.setStyleSheet("color: #4ade80; font-weight: 700; font-size: 13px;")
                        self.lbl_val_details.setText("No structural anomalies detected. Syntax and measure durations verified.")
                        self.card_val.setStyleSheet("CardWidget { border-left: 4px solid #238636; }")
                    else:
                        self.lbl_val_status.setText(f"Staff {selected_crop_idx + 1} — Score: {score_val}% (ANOMALY DETECTED)")
                        self.lbl_val_status.setStyleSheet("color: #f87171; font-weight: 700; font-size: 13px;")
                        self.lbl_val_details.setText(" • " + "\n • ".join(rep.anomalies) if rep.anomalies else "Parsing issue")
                        self.card_val.setStyleSheet("CardWidget { border-left: 4px solid #da3633; }")

                    # Render vector SVG using Verovio
                    svg_str = ""
                    if raw_kern:
                        svg_str = self.bridge.render_svg(raw_kern, scale=80)
                    if svg_str:
                        clean_svg = self.bridge.flatten_svg(svg_str)
                        self.canvas_score.load_svg_content(clean_svg)
                    else:
                        self.canvas_score.clear_view()

                    # Populate code editor
                    code_disp = f"=== ABC NOTATION ===\n{abc_text}\n\n=== HUMDRUM KERN ===\n{raw_kern}"
                    self.text_editor.setPlainText(code_disp)
                    self.lbl_right_title.setText(f"PAGE {p_num} (STAFF {selected_crop_idx + 1}): VEROVIO VECTOR SCORE & ABC")
                else:
                    self.lbl_val_status.setText(f"Staff {selected_crop_idx + 1} — Awaiting OMR recognition")
                    self.lbl_val_status.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 13px;")
                    self.lbl_val_details.setText("Staff crop extracted. Run OMR pipeline to decode notes.")
                    self.card_val.setStyleSheet("")
                    self.canvas_score.clear_view()
                    self.text_editor.setPlainText("Awaiting Transcoda OMR recognition.")
                    self.lbl_right_title.setText(f"PAGE {p_num}: AWAITING RECOGNITION")
            else:
                src_page = debug_file if debug_file.is_file() else mask_file
                if src_page and src_page.is_file():
                    self.canvas_left.load_file(str(src_page))
                else:
                    self.canvas_left.clear_view()
                self.lbl_left_title.setText(f"PAGE {p_num}: NO STAVES DETECTED")
                self.lbl_val_status.setText("No musical staves on this page")
                self.lbl_val_status.setStyleSheet("color: #94a3b8;")
                self.lbl_val_details.setText("Standard text/prose page.")
                self.card_val.setStyleSheet("")
                self.canvas_score.clear_view()
                self.text_editor.setPlainText(f"No musical staves detected on page {p_num}.")
                self.lbl_right_title.setText(f"PAGE {p_num}: NO NOTES")

        # -------------------------------------------------------------
        # Mode 4: Markdown
        # -------------------------------------------------------------
        elif "4. Markdown" in cur_text:
            self.right_stack.setCurrentIndex(2)
            self.btn_copy.setVisible(True)
            self.lbl_left_title.setText(f"PAGE {p_num}: PAGE SCAN")

            src_page = debug_file if debug_file.is_file() else mask_file
            if src_page and src_page.is_file():
                self.canvas_left.load_file(str(src_page))
            else:
                self.canvas_left.clear_view()

            if final_md_file.is_file():
                self.text_editor_md.setPlainText(final_md_file.read_text(encoding="utf-8"))
                self.lbl_right_title.setText(f"PAGE {p_num}: ASSEMBLED MARKDOWN DOCUMENT")
            elif raw_md_file.is_file():
                self.text_editor_md.setPlainText(raw_md_file.read_text(encoding="utf-8"))
                self.lbl_right_title.setText(f"PAGE {p_num}: RAW VLM DRAFT (WITHOUT NOTES)")
            else:
                self.text_editor_md.setPlainText(
                    f"Markdown document for page {p_num} has not been generated yet.\n"
                    "Page is awaiting VLM OCR (Qwen 3.5) or final assembly."
                )
                self.lbl_right_title.setText(f"PAGE {p_num}: AWAITING PROCESSING")

    def _copy_code(self) -> None:
        cur_text = self.mode_segmented.currentItem().text() if self.mode_segmented.currentItem() else ""
        txt = self.text_editor_md.toPlainText() if "4. Markdown" in cur_text else self.text_editor.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            InfoBar.success(
                title="Copied",
                content="Code copied to clipboard.",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self,
            )
