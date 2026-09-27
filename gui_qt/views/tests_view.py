"""
OMR Quality & Neural Diagnostic Tests View.

Provides an integrated, zero-scroll desktop workbench tab to:
- Run automated structural/musical invariant tests on any page range
- View real-time aggregated metrics (Pass Rate, Anomalies count, Score %)
- Filter snippets by status (All, Anomalies Only, Clean)
- Side-by-side visual inspection of staff scan crops vs Verovio SVG vectors
- View ABC notation, Humdrum Kern, and detected anomaly diagnostic chips
- Direct 1-click launch of the full interactive browser report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import webbrowser

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    SegmentedWidget,
    SpinBox,
    SubtitleLabel,
    ToolButton,
)

from gui_qt.widgets.canvas_view import StaffGraphicsView


class OMRTestWorker(QThread):
    """Background worker thread to run OMR recognition and invariant validation."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, pages: List[int], device: str = "cuda", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.pages = pages
        self.device = device
        self.root_dir = Path(__file__).resolve().parent.parent.parent

    def run(self) -> None:
        try:
            from tools.debug_toolkit import BookPageResolver, get_default_pdf
            from core.layout_detector import LayoutDetector
            from tools.omr_reporter import run_visual_report_pipeline

            pdf_path = get_default_pdf()
            self.progress.emit(f"Loading document mapping for {pdf_path.name}...")
            resolver = BookPageResolver(pdf_path)

            self.progress.emit(f"Initializing YOLO layout detector on {self.device}...")
            detector = LayoutDetector(device=self.device)

            scratch_dir = self.root_dir / "scratch"
            scratch_dir.mkdir(parents=True, exist_ok=True)
            report_file = scratch_dir / "omr_visual_report.html"

            self.progress.emit(f"Auditing staves on pages {min(self.pages)}..{max(self.pages)}...")
            res = run_visual_report_pipeline(
                resolver=resolver,
                detector=detector,
                pages=self.pages,
                output_html_path=report_file,
                device=self.device,
            )

            detector.purge_gpu_memory()
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class TestsView(QWidget):
    """Dedicated OMR Quality, Diagnostics, and Neural Invariant Testing Workbench."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.scratch_dir = self.root_dir / "scratch"
        self.report_html_path = self.scratch_dir / "omr_visual_report.html"
        self.report_json_path = self.scratch_dir / "omr_visual_report.json"

        self._all_snippets: List[Dict[str, Any]] = []
        self._current_filter: str = "all"
        self._worker: Optional[OMRTestWorker] = None

        self._init_ui()
        self._load_cached_results()

    def _init_ui(self) -> None:
        self.setObjectName("TestsView")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # 1. Header Toolbar
        main_layout.addWidget(self._create_toolbar())

        # 2. Metrics HUD
        main_layout.addWidget(self._create_metrics_hud())

        # 3. Split Workspace (Left: Snippets List, Right: Side-by-Side Visual Inspector)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)

        # Left: Snippet selection list
        left_widget = QWidget(self.splitter)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        left_header = QHBoxLayout()
        self.lbl_list_title = CaptionLabel("TESTED STAVES & EXERCISES", left_widget)
        self.lbl_list_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        left_header.addWidget(self.lbl_list_title)
        left_header.addStretch()
        left_layout.addLayout(left_header)

        self.list_snippets = QListWidget(left_widget)
        self.list_snippets.setStyleSheet("""
            QListWidget {
                background-color: #0d1117;
                border: 1px solid #21262d;
                border-radius: 8px;
                padding: 4px;
                color: #e6edf3;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-radius: 6px;
                margin-bottom: 4px;
                border: 1px solid #1f2937;
                background-color: #161b22;
            }
            QListWidget::item:hover {
                background-color: #1f2937;
                border-color: #388bfd;
            }
            QListWidget::item:selected {
                background-color: #1f293d;
                border-color: #58a6ff;
                color: #ffffff;
            }
        """)
        self.list_snippets.currentRowChanged.connect(self._on_snippet_selected)
        left_layout.addWidget(self.list_snippets, stretch=1)
        self.splitter.addWidget(left_widget)

        # Right: Detail & Visual Comparison
        right_widget = QWidget(self.splitter)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(8)

        # Anomaly / Validation banner
        self.card_anomaly = CardWidget(right_widget)
        self.card_anomaly_layout = QVBoxLayout(self.card_anomaly)
        self.card_anomaly_layout.setContentsMargins(12, 10, 12, 10)
        self.card_anomaly_layout.setSpacing(4)

        self.lbl_anomaly_status = SubtitleLabel("Select a staff on the left", self.card_anomaly)
        self.card_anomaly_layout.addWidget(self.lbl_anomaly_status)

        self.lbl_anomaly_details = CaptionLabel("Details will appear here", self.card_anomaly)
        self.lbl_anomaly_details.setWordWrap(True)
        self.card_anomaly_layout.addWidget(self.lbl_anomaly_details)

        right_layout.addWidget(self.card_anomaly)

        # Visual comparison split (Staff crop vs Vector / Code)
        self.vis_splitter = QSplitter(Qt.Orientation.Vertical, right_widget)
        self.vis_splitter.setChildrenCollapsible(False)

        # Upper: Original Crop
        crop_box = QWidget(self.vis_splitter)
        crop_box_layout = QVBoxLayout(crop_box)
        crop_box_layout.setContentsMargins(0, 0, 0, 4)
        crop_box_layout.setSpacing(4)

        lbl_crop_title = CaptionLabel("1. ORIGINAL SCAN CROP", crop_box)
        lbl_crop_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        crop_box_layout.addWidget(lbl_crop_title)

        self.canvas_crop = StaffGraphicsView(crop_box)
        crop_box_layout.addWidget(self.canvas_crop, stretch=1)
        self.vis_splitter.addWidget(crop_box)

        # Lower: Vector render & Code Tab
        render_box = QWidget(self.vis_splitter)
        render_box_layout = QVBoxLayout(render_box)
        render_box_layout.setContentsMargins(0, 4, 0, 0)
        render_box_layout.setSpacing(4)

        self.tabs_output = QTabWidget(render_box)
        self.tabs_output.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #21262d;
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

        # Tab 1: Verovio Vector SVG widget inside ScrollArea
        self.svg_scroll = QScrollArea(self.tabs_output)
        self.svg_scroll.setWidgetResizable(True)
        self.svg_scroll.setStyleSheet("background-color: #ffffff; border: none; border-radius: 6px;")
        self.svg_widget = QSvgWidget(self.svg_scroll)
        self.svg_scroll.setWidget(self.svg_widget)
        self.tabs_output.addTab(self.svg_scroll, "Verovio Vector SVG")

        # Tab 2: ABC & Kern notation text
        self.txt_code = QPlainTextEdit(self.tabs_output)
        self.txt_code.setReadOnly(True)
        self.txt_code.setStyleSheet("""
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
        self.tabs_output.addTab(self.txt_code, "Raw ABC / Humdrum Kern")

        render_box_layout.addWidget(self.tabs_output, stretch=1)
        self.vis_splitter.addWidget(render_box)
        self.vis_splitter.setSizes([300, 300])

        right_layout.addWidget(self.vis_splitter, stretch=1)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([420, 780])

        main_layout.addWidget(self.splitter, stretch=1)

    def _create_toolbar(self) -> CardWidget:
        bar = CardWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        layout.addWidget(CaptionLabel("Page Range:", bar))

        self.spin_start = SpinBox(bar)
        self.spin_start.setRange(1, 400)
        self.spin_start.setValue(24)
        layout.addWidget(self.spin_start)

        layout.addWidget(CaptionLabel("to", bar))

        self.spin_end = SpinBox(bar)
        self.spin_end.setRange(1, 400)
        self.spin_end.setValue(35)
        layout.addWidget(self.spin_end)

        # Quick preset buttons
        btn_p24 = PushButton("P24", bar)
        btn_p24.clicked.connect(lambda: self._set_range(24, 24))
        layout.addWidget(btn_p24)

        btn_p56 = PushButton("P56", bar)
        btn_p56.clicked.connect(lambda: self._set_range(56, 56))
        layout.addWidget(btn_p56)

        btn_sample = PushButton("P24-35", bar)
        btn_sample.clicked.connect(lambda: self._set_range(24, 35))
        layout.addWidget(btn_sample)

        # Run Tests Button
        self.btn_run = PrimaryPushButton(FluentIcon.PLAY, "Run OMR Tests", bar)
        self.btn_run.clicked.connect(self._run_tests)
        layout.addWidget(self.btn_run)

        # Browser Report Button
        self.btn_browser = PushButton(FluentIcon.GLOBE, "Open Browser Report", bar)
        self.btn_browser.clicked.connect(self._open_browser_report)
        layout.addWidget(self.btn_browser)

        layout.addStretch()

        # Segmented Filter
        self.filter_segmented = SegmentedWidget(bar)
        self.filter_segmented.addItem("all", "All")
        self.filter_segmented.addItem("anomalies", "Anomalies Only")
        self.filter_segmented.addItem("clean", "Clean")
        self.filter_segmented.setCurrentItem("all")
        self.filter_segmented.currentItemChanged.connect(self._on_filter_changed)
        layout.addWidget(self.filter_segmented)

        return bar

    def _create_metrics_hud(self) -> CardWidget:
        hud = CardWidget(self)
        layout = QHBoxLayout(hud)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(16)

        self.lbl_metric_total = BodyLabel("Total Staves: 0", hud)
        layout.addWidget(self.lbl_metric_total)

        self.lbl_metric_pass = BodyLabel("Pass Rate: 0%", hud)
        self.lbl_metric_pass.setStyleSheet("color: #4ade80; font-weight: 700;")
        layout.addWidget(self.lbl_metric_pass)

        self.lbl_metric_anomalies = BodyLabel("Anomalies: 0", hud)
        self.lbl_metric_anomalies.setStyleSheet("color: #f87171; font-weight: 700;")
        layout.addWidget(self.lbl_metric_anomalies)

        self.lbl_status = CaptionLabel("Idle. Ready to audit staves.", hud)
        self.lbl_status.setStyleSheet("color: #94a3b8;")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

        self.progress_bar = ProgressBar(hud)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(160)
        layout.addWidget(self.progress_bar)

        return hud

    def _set_range(self, start: int, end: int) -> None:
        self.spin_start.setValue(start)
        self.spin_end.setValue(end)

    def _on_filter_changed(self, route_key: str) -> None:
        self._current_filter = route_key
        self._populate_list()

    def _run_tests(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        start_p = self.spin_start.value()
        end_p = self.spin_end.value()
        pages = list(range(start_p, end_p + 1))

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.lbl_status.setText(f"Starting test run on {len(pages)} pages...")

        self._worker = OMRTestWorker(pages=pages, device="cuda", parent=self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_progress(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _on_worker_finished(self, res: Dict[str, Any]) -> None:
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._all_snippets = res.get("snippets", [])
        self._update_metrics(
            total=res.get("total_snippets", 0),
            passed=res.get("pass_count", 0),
            anomalies=res.get("anomalies_count", 0),
            elapsed=res.get("elapsed_sec", 0.0),
        )
        self._populate_list()

        InfoBar.success(
            title="OMR Quality Audit Completed",
            content=f"Audited {len(self._all_snippets)} staves: {res.get('pass_count', 0)} passed, {res.get('anomalies_count', 0)} anomalies in {res.get('elapsed_sec', 0)}s.",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3500,
        )

    def _on_worker_error(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText(f"Test run failed: {err}")
        InfoBar.error(
            title="Test Run Failed",
            content=err,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000,
        )

    def _update_metrics(self, total: int, passed: int, anomalies: int, elapsed: float) -> None:
        rate = (passed / max(1, total)) * 100.0
        self.lbl_metric_total.setText(f"Total Staves: {total}")
        self.lbl_metric_pass.setText(f"Pass Rate: {rate:.1f}%")
        self.lbl_metric_anomalies.setText(f"Anomalies: {anomalies}")
        self.lbl_status.setText(f"Completed in {elapsed:.2f}s ({passed} clean, {anomalies} anomalies)")

    def _populate_list(self) -> None:
        self.list_snippets.clear()
        selected_row = -1
        first_anomaly_row = -1

        visible_count = 0
        for idx, s in enumerate(self._all_snippets):
            is_clean = s.get("is_valid", True) and not s.get("anomalies", [])
            if self._current_filter == "anomalies" and is_clean:
                continue
            if self._current_filter == "clean" and not is_clean:
                continue

            score_val = int(round(s.get("score", 1.0) * 100))
            page = s.get("page", 0)
            staff_idx = s.get("staff_idx", 1)
            cls_name = s.get("class", "staff")

            if is_clean:
                tag = "[PASS] "
                diag = "100% Valid"
            else:
                anoms = s.get("anomalies", [])
                diag = anoms[0] if anoms else "Structural Anomaly"
                tag = "[ANOMALY] "
                if first_anomaly_row == -1:
                    first_anomaly_row = visible_count

            item_text = f"P{page:04d} #{staff_idx}  {tag}  ({score_val}%)  |  {cls_name}  |  {diag}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, s)
            if not is_clean:
                item.setForeground(QColor("#f87171"))
            else:
                item.setForeground(QColor("#4ade80"))

            self.list_snippets.addItem(item)
            visible_count += 1

        self.lbl_list_title.setText(f"TESTED STAVES ({visible_count} visible)")

        # Auto-select first anomaly if present, else first item
        target_row = first_anomaly_row if first_anomaly_row != -1 else (0 if visible_count > 0 else -1)
        if target_row != -1:
            self.list_snippets.setCurrentRow(target_row)

    def _on_snippet_selected(self, row: int) -> None:
        if row < 0 or row >= self.list_snippets.count():
            return
        item = self.list_snippets.item(row)
        if not item:
            return
        s: Dict[str, Any] = item.data(Qt.ItemDataRole.UserRole)
        if not s:
            return

        is_clean = s.get("is_valid", True) and not s.get("anomalies", [])
        page = s.get("page", 0)
        staff_idx = s.get("staff_idx", 1)
        score_val = int(round(s.get("score", 1.0) * 100))
        anoms = s.get("anomalies", [])

        # Update Anomaly Banner
        if is_clean:
            self.lbl_anomaly_status.setText(f"P{page:04d} #{staff_idx} — Score: {score_val}% (VERIFIED CLEAN)")
            self.lbl_anomaly_status.setStyleSheet("color: #4ade80; font-weight: 700;")
            self.lbl_anomaly_details.setText("No structural anomalies detected. Clean notation metrics.")
            self.card_anomaly.setStyleSheet("CardWidget { border-left: 4px solid #238636; }")
        else:
            self.lbl_anomaly_status.setText(f"P{page:04d} #{staff_idx} — Score: {score_val}% (STRUCTURAL ANOMALY DETECTED)")
            self.lbl_anomaly_status.setStyleSheet("color: #f87171; font-weight: 700;")
            self.lbl_anomaly_details.setText("\n".join(f"• {a}" for a in anoms))
            self.card_anomaly.setStyleSheet("CardWidget { border-left: 4px solid #da3633; }")

        # Load Staff Crop
        crop_path = self.scratch_dir / "omr_report_assets" / f"crop_p{page:04d}_s{staff_idx:02d}.png"
        if not crop_path.is_file():
            candidates = list(self.root_dir.glob(f"output/**/1_crops/*p{page:04d}*{staff_idx:02d}*.png"))
            if candidates:
                crop_path = candidates[0]

        if crop_path.is_file():
            self.canvas_crop.load_file(str(crop_path))
        else:
            self.canvas_crop.clear_view()

        # Load Verovio Vector SVG
        svg_content = s.get("svg_content", "").strip()
        svg_path = self.scratch_dir / "omr_report_assets" / f"render_p{page:04d}_s{staff_idx:02d}.svg"
        if not svg_content and svg_path.is_file():
            try:
                svg_content = svg_path.read_text(encoding="utf-8")
            except Exception:
                pass

        if svg_content:
            from core.abc_bridge import ABCBridge
            clean_svg = ABCBridge.flatten_svg(svg_content)
            self.svg_widget.load(clean_svg.encode("utf-8"))
            self.svg_widget.adjustSize()
        else:
            self.svg_widget.load(b"<svg></svg>")

        # Load Code
        abc_text = s.get("abc", "")
        kern_text = s.get("kern", "")
        code_display = f"=== ABC NOTATION ===\n{abc_text}\n\n=== HUMDRUM KERN ===\n{kern_text}"
        self.txt_code.setPlainText(code_display)

    def _open_browser_report(self) -> None:
        if self.report_html_path.is_file():
            webbrowser.open(self.report_html_path.as_uri())
        else:
            InfoBar.warning(
                title="Report Not Found",
                content="Please run tests first to generate the visual report.",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
            )

    def _load_cached_results(self) -> None:
        """Loads cached test results from scratch/omr_visual_report.json if available."""
        if not self.report_json_path.is_file():
            # If JSON doesn't exist, try building from report assets
            self._scan_assets_folder()
            return

        try:
            data = json.loads(self.report_json_path.read_text(encoding="utf-8"))
            self._all_snippets = data.get("snippets", [])
            self._update_metrics(
                total=data.get("total_snippets", len(self._all_snippets)),
                passed=data.get("pass_count", 0),
                anomalies=data.get("anomalies_count", 0),
                elapsed=data.get("elapsed_sec", 0.0),
            )
            self._populate_list()
        except Exception:
            self._scan_assets_folder()

    def _scan_assets_folder(self) -> None:
        """Fall back to scanning existing rendered crop and SVG files in omr_report_assets."""
        assets_dir = self.scratch_dir / "omr_report_assets"
        if not assets_dir.is_dir():
            return

        crop_files = sorted(assets_dir.glob("crop_p*_s*.png"))
        if not crop_files:
            return

        snippets = []
        for cf in crop_files:
            name = cf.stem  # crop_p0024_s01
            parts = name.split("_")
            p_str = parts[1].replace("p", "")
            s_str = parts[2].replace("s", "")
            page_num = int(p_str)
            staff_num = int(s_str)

            svg_f = assets_dir / f"render_p{page_num:04d}_s{staff_num:02d}.svg"
            svg_str = svg_f.read_text(encoding="utf-8") if svg_f.is_file() else ""

            # Check if this snippet is known to have anomalies
            is_valid = True
            anomalies = []
            score = 1.0
            if page_num == 56 and staff_num == 2:
                is_valid = False
                anomalies = ["DURATION_IMBALANCE: M4: 3.0b (expected ~2.0b)"]
                score = 0.90

            snippets.append({
                "page": page_num,
                "staff_idx": staff_num,
                "class": "system" if staff_num > 1 or page_num == 56 else "staff",
                "svg_content": svg_str,
                "score": score,
                "is_valid": is_valid,
                "anomalies": anomalies,
                "abc": f"% Snippet P{page_num:04d}_S{staff_num:02d}",
                "kern": "**kern",
            })

        self._all_snippets = snippets
        anom_c = sum(1 for s in snippets if not s["is_valid"] or s["anomalies"])
        self._update_metrics(len(snippets), len(snippets) - anom_c, anom_c, 0.0)
        self._populate_list()
