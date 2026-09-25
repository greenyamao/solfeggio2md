"""
Pipeline Dashboard View.

Primary operational workspace featuring:
- Top Action Command Bar (Start, Pause, Stop, Open Folder)
- Active Book & Queue Tracker
- 4-Phase Stepper Card
- Interactive High-Performance Live Monitor (StaffGraphicsView)
- Real-Time Hardware & VLM Telemetry Ribbon
- Monospace Live Neural Console
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    FluentIcon,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    SubtitleLabel,
    TableWidget,
    ToolButton,
)

from gui_qt.ipc.shared_frame import SharedFrameReader
from gui_qt.ipc.worker_process import MLPipelineProcessClient
from gui_qt.widgets.canvas_view import StaffGraphicsView
from gui_qt.widgets.live_console import LiveConsoleWidget
from gui_qt.widgets.stage_progress import StageProgressWidget
from gui_qt.widgets.telemetry_hud import TelemetryHUDRibbon


class PipelineView(QWidget):
    """Main pipeline execution dashboard."""

    def __init__(
        self,
        worker_client: MLPipelineProcessClient,
        frame_reader: SharedFrameReader,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.worker_client = worker_client
        self.frame_reader = frame_reader

        self._current_image_path: str = ""
        self._init_ui()
        self._setup_timer()

    def _init_ui(self) -> None:
        self.setObjectName("PipelineView")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # 1. Top Command Bar
        main_layout.addWidget(self._create_command_bar())

        # 2. Main Horizontal Splitter (Left: Queue/Stages, Right: Monitor/Console)
        h_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        h_splitter.setChildrenCollapsible(False)

        # Left Column: Queue & Stages
        left_widget = QWidget(h_splitter)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # Active Book Card
        self.book_card = self._create_book_card()
        left_layout.addWidget(self.book_card)

        # 4-Phase Stepper
        self.stage_progress = StageProgressWidget(left_widget)
        left_layout.addWidget(self.stage_progress)

        # File Queue Table
        self.queue_card = self._create_queue_card()
        left_layout.addWidget(self.queue_card, stretch=1)

        h_splitter.addWidget(left_widget)

        # Right Column: Live Monitor & Console
        right_splitter = QSplitter(Qt.Orientation.Vertical, h_splitter)
        right_splitter.setChildrenCollapsible(False)

        # Top Right: Live Canvas & Telemetry
        monitor_container = QWidget(right_splitter)
        mon_layout = QVBoxLayout(monitor_container)
        mon_layout.setContentsMargins(8, 0, 0, 0)
        mon_layout.setSpacing(8)

        # Horizontal Splitter between Live Visual Canvas and Live Markdown/ABC Text
        self.mon_splitter = QSplitter(Qt.Orientation.Horizontal, monitor_container)
        self.mon_splitter.setChildrenCollapsible(False)

        # Left: Live Graphics Canvas
        self.canvas_view = StaffGraphicsView(self.mon_splitter)
        self.mon_splitter.addWidget(self.canvas_view)

        # Right: Live Text Card (Markdown / ABC)
        self.text_preview_card = self._create_text_preview_card()
        self.mon_splitter.addWidget(self.text_preview_card)

        # Proportions: 55% visual canvas, 45% live text
        self.mon_splitter.setSizes([450, 370])

        # Monitor Header Card (instantiated after canvas_view and text_preview_card)
        mon_header = self._create_monitor_header()
        mon_layout.addWidget(mon_header)
        mon_layout.addWidget(self.mon_splitter, stretch=1)


        # Telemetry Ribbon
        self.telemetry_hud = TelemetryHUDRibbon(monitor_container)
        mon_layout.addWidget(self.telemetry_hud)

        right_splitter.addWidget(monitor_container)

        # Bottom Right: Live Console
        console_container = QWidget(right_splitter)
        con_layout = QVBoxLayout(console_container)
        con_layout.setContentsMargins(8, 6, 0, 0)
        con_layout.setSpacing(0)

        self.live_console = LiveConsoleWidget(console_container)
        con_layout.addWidget(self.live_console)

        right_splitter.addWidget(console_container)

        # Set default splitter proportions (65% monitor, 35% console)
        right_splitter.setSizes([460, 240])

        h_splitter.addWidget(right_splitter)
        # Set default horizontal proportions (38% left, 62% right)
        h_splitter.setSizes([380, 620])

        main_layout.addWidget(h_splitter, stretch=1)

    def _create_command_bar(self) -> QWidget:
        bar = CardWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Action Buttons
        self.btn_start = PrimaryPushButton(FluentIcon.PLAY, "Start Pipeline", bar)
        self.btn_start.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.btn_start)

        self.btn_pause = PushButton(FluentIcon.PAUSE, "Pause", bar)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        layout.addWidget(self.btn_pause)

        self.btn_stop = PushButton(FluentIcon.POWER_BUTTON, "Stop", bar)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.btn_stop)

        self.btn_folder = PushButton(FluentIcon.FOLDER, "Folder in/", bar)
        self.btn_folder.clicked.connect(self._on_open_folder_clicked)
        layout.addWidget(self.btn_folder)

        layout.addStretch()

        # Status text in command bar
        self.lbl_bar_status = QLabel("Ready", bar)
        self.lbl_bar_status.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 12px;")
        layout.addWidget(self.lbl_bar_status)

        return bar

    def _create_book_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        caption = CaptionLabel("CURRENT BOOK", card)
        caption.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
        header_layout.addWidget(caption)
        header_layout.addStretch()

        self.lbl_book_pages = CaptionLabel("0 / 0 pages", card)
        self.lbl_book_pages.setStyleSheet("color: #38bdf8; font-weight: 600;")
        header_layout.addWidget(self.lbl_book_pages)
        layout.addLayout(header_layout)

        self.lbl_book_title = QLabel("Queue awaiting command", card)
        self.lbl_book_title.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: 700;")
        self.lbl_book_title.setWordWrap(True)
        layout.addWidget(self.lbl_book_title)

        self.book_progress_bar = ProgressBar(card)
        self.book_progress_bar.setValue(0)
        self.book_progress_bar.setFixedHeight(6)
        layout.addWidget(self.book_progress_bar)

        return card

    def _create_queue_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        caption = CaptionLabel("FILE QUEUE (in/)", card)
        caption.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
        layout.addWidget(caption)

        self.queue_table = TableWidget(card)
        self.queue_table.setColumnCount(3)
        self.queue_table.setHorizontalHeaderLabels(["File", "Pages", "Status"])
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)

        layout.addWidget(self.queue_table)
        return card

    def _create_monitor_header(self) -> CardWidget:
        card = CardWidget(self)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        self.lbl_mon_title = CaptionLabel("LIVE MONITOR OF CURRENT STEP", card)
        self.lbl_mon_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700;")
        layout.addWidget(self.lbl_mon_title)

        layout.addStretch()

        self.badge_stage = QLabel("PHASE: IDLE", card)
        self.badge_stage.setStyleSheet("""
            QLabel {
                background-color: #1e293b;
                color: #38bdf8;
                font-weight: 700;
                font-size: 10px;
                padding: 2px 8px;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.badge_stage)

        self.btn_fit = ToolButton(FluentIcon.ZOOM_IN, card)
        self.btn_fit.setToolTip("Fit to window")
        self.btn_fit.clicked.connect(self.canvas_view.fit_to_view)
        layout.addWidget(self.btn_fit)

        self.btn_toggle_text = ToolButton(FluentIcon.DOCUMENT, card)
        self.btn_toggle_text.setToolTip("Show / hide recognized text panel")
        self.btn_toggle_text.clicked.connect(self._toggle_text_panel)
        layout.addWidget(self.btn_toggle_text)

        return card

    def _create_text_preview_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        hdr_layout = QHBoxLayout()
        self.lbl_text_title = CaptionLabel("RECOGNIZED TEXT (MARKDOWN / ABC)", card)
        self.lbl_text_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        hdr_layout.addWidget(self.lbl_text_title)

        hdr_layout.addStretch()

        self.badge_text_status = QLabel("IDLE", card)
        self.badge_text_status.setStyleSheet("""
            QLabel {
                background-color: #1e293b;
                color: #94a3b8;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10px;
                font-weight: 700;
                padding: 2px 8px;
                border-radius: 4px;
            }
        """)
        hdr_layout.addWidget(self.badge_text_status)

        self.btn_copy_text = ToolButton(FluentIcon.COPY, card)
        self.btn_copy_text.setToolTip("Copy recognized text")
        self.btn_copy_text.clicked.connect(self._copy_live_text)
        hdr_layout.addWidget(self.btn_copy_text)

        layout.addLayout(hdr_layout)

        self.text_preview = QPlainTextEdit(card)
        self.text_preview.setReadOnly(True)
        self.text_preview.setPlaceholderText("Real-time generated Markdown text or OMR code will appear here...")
        self.text_preview.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0b0f17;
                color: #e2e8f0;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 12px;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 8px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.text_preview, stretch=1)
        return card

    def _toggle_text_panel(self) -> None:
        vis = not self.text_preview_card.isVisible()
        self.text_preview_card.setVisible(vis)

    def _copy_live_text(self) -> None:
        txt = self.text_preview.toPlainText()
        if txt:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(txt)

    def _update_live_text(self, data: Dict[str, Any]) -> None:
        stage = data.get("stage", "phase3")
        is_comp = data.get("completed", False)
        chunk = data.get("chunk")
        full_text = data.get("text", "")
        page = data.get("page", 0)

        if stage == "phase2":
            self.lbl_text_title.setText("RECOGNIZED NOTES (ABC CODE)")
            self.lbl_text_title.setStyleSheet("color: #facc15; font-weight: 700; font-size: 11px;")
            self.text_preview.setPlainText(full_text)
            self.badge_text_status.setText("OMR DONE")
            self.badge_text_status.setStyleSheet("""
                QLabel {
                    background-color: #854d0e;
                    color: #fef08a;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
            """)
            return

        if stage == "phase3_analyzing":
            # Keep previous page image, title, and text visible while vision encoder is deciphering page
            self.badge_text_status.setText(f"ANALYZING PAGE {page}...")
            self.badge_text_status.setStyleSheet("""
                QLabel {
                    background-color: #431407;
                    color: #fed7aa;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
            """)
            return

        if stage == "phase3_disconnected":
            self.badge_text_status.setText("SERVER DISCONNECTED")
            self.badge_text_status.setStyleSheet("""
                QLabel {
                    background-color: #7f1d1d;
                    color: #fca5a5;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
            """)
            return

        # Phase 3: VLM Markdown
        img_path = data.get("image_path")
        first_chunk = data.get("first_chunk", False)

        # Synchronize visual canvas and titles strictly when new page tokens begin
        is_new_page = first_chunk or (page and page != self._active_text_page)
        if is_new_page:
            self._active_text_page = page
            self.text_preview.clear()
            title_str = f"RECOGNIZED TEXT (MARKDOWN • PAGE {page})" if page else "RECOGNIZED TEXT (MARKDOWN)"
            self.lbl_text_title.setText(title_str)
            self.lbl_text_title.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
            if img_path and os.path.isfile(img_path):
                if img_path != self._current_image_path:
                    self._current_image_path = img_path
                    self.canvas_view.load_file(img_path)
                self.lbl_mon_title.setText(f"LIVE MONITOR: Page {page} — VLM masked page")
        elif img_path and os.path.isfile(img_path) and img_path != self._current_image_path:
            self._current_image_path = img_path
            self.canvas_view.load_file(img_path)
            self.lbl_mon_title.setText(f"LIVE MONITOR: Page {page} — VLM masked page")

        if is_comp:
            self.text_preview.setPlainText(full_text)
            self.badge_text_status.setText("DONE")
            self.badge_text_status.setStyleSheet("""
                QLabel {
                    background-color: #14532d;
                    color: #86efac;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
            """)
        else:
            if chunk is not None:
                self.text_preview.moveCursor(QTextCursor.MoveOperation.End)
                self.text_preview.insertPlainText(chunk)
                self.text_preview.moveCursor(QTextCursor.MoveOperation.End)
            elif full_text:
                self.text_preview.setPlainText(full_text)
            self.badge_text_status.setText("GENERATING...")
            self.badge_text_status.setStyleSheet("""
                QLabel {
                    background-color: #1e3a8a;
                    color: #93c5fd;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 10px;
                    font-weight: 700;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
            """)

    def _setup_timer(self) -> None:
        """Polls status queue and shared memory frame buffer every 100 ms."""
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def _on_tick(self) -> None:
        # 1. Drain messages from worker
        messages = self.worker_client.poll_messages(100)
        for msg in messages:
            mtype = msg.get("type")
            if mtype == "log":
                data = msg.get("data", {})
                self.live_console.append_log(data)

            elif mtype == "frame":
                meta = msg.get("meta", {})
                path = meta.get("path", "")
                title = meta.get("title", "")
                stage = meta.get("stage", "")

                # For Phase 1 & 2, update canvas immediately. For Phase 3, canvas updates on token generation!
                if stage != "phase3":
                    if title and self.lbl_mon_title.text() != f"LIVE MONITOR: {title}":
                        self.lbl_mon_title.setText(f"LIVE MONITOR: {title}")
                    if path and path != self._current_image_path and os.path.isfile(path):
                        self._current_image_path = path
                        self.canvas_view.load_file(path)

            elif mtype == "text_update":
                data = msg.get("data", {})
                self._update_live_text(data)

            elif mtype == "status":
                metrics = msg.get("metrics", {})
                self._update_from_metrics(metrics)

                q_summary = msg.get("queue_summary", {})
                self._update_queue_table(q_summary.get("queue", []))

        # 2. Check for zero-copy frame update in shared memory
        qimg, fid = self.frame_reader.read_qimage()
        if qimg is not None:
            self.canvas_view.set_qimage(qimg)

    def _update_from_metrics(self, metrics: Dict[str, Any]) -> None:
        is_running = metrics.get("is_running", False)
        is_paused = metrics.get("is_paused", False)

        if is_running:
            if is_paused:
                if self.lbl_bar_status.text() != "Paused":
                    self.lbl_bar_status.setText("Paused")
                    self.lbl_bar_status.setStyleSheet("color: #fb923c; font-weight: 600;")
                if self.btn_pause.text() != "Resume":
                    self.btn_pause.setText("Resume")
            else:
                if self.lbl_bar_status.text() != "Running...":
                    self.lbl_bar_status.setText("Running...")
                    self.lbl_bar_status.setStyleSheet("color: #4ade80; font-weight: 600;")
                if self.btn_pause.text() != "Pause":
                    self.btn_pause.setText("Pause")
        else:
            if self.lbl_bar_status.text() != "Stopped / Ready":
                self.lbl_bar_status.setText("Stopped / Ready")
                self.lbl_bar_status.setStyleSheet("color: #38bdf8; font-weight: 600;")

        # Active book
        book_name = metrics.get("current_book_name", "")
        if book_name and self.lbl_book_title.text() != book_name:
            self.lbl_book_title.setText(book_name)
        phase_name = metrics.get("current_phase_name", "Idle")
        phase_str = f"PHASE: {phase_name.upper()}"
        if self.badge_stage.text() != phase_str:
            self.badge_stage.setText(phase_str)

        pct = int(float(metrics.get("book_progress_pct", 0.0)))
        if self.book_progress_bar.value() != pct:
            self.book_progress_bar.setValue(pct)

        # 4 phases
        phases = metrics.get("phase_progress", {})
        if phases:
            self.stage_progress.update_phases(phases)

        # Telemetry
        self.telemetry_hud.update_metrics(metrics)

        # Image from metrics if not already loaded (Phase 1 & 2 only, or initial idle state)
        phase_str = str(metrics.get("current_phase_name", "")).lower()
        is_phase3_active = ("vlm" in phase_str or "phase 3" in phase_str or "phase3" in phase_str)
        if not is_phase3_active or not self._current_image_path:
            img_path = metrics.get("current_page_image_path", "")
            if img_path and img_path != self._current_image_path and os.path.isfile(img_path):
                self._current_image_path = img_path
                self.canvas_view.load_file(img_path)

        # Text fallback from metrics if preview is empty
        vlm_text = metrics.get("current_vlm_text", "")
        if vlm_text and not self.text_preview.toPlainText():
            self.text_preview.setPlainText(vlm_text)

    def _update_queue_table(self, queue_items: List[Dict[str, Any]]) -> None:
        sig = tuple((it.get("name", ""), it.get("pages", 0), it.get("status", "")) for it in queue_items)
        if getattr(self, "_last_queue_sig", None) == sig:
            return
        self._last_queue_sig = sig

        self.queue_table.setRowCount(len(queue_items))
        for row, item in enumerate(queue_items):
            name = item.get("name", "")
            pages = str(item.get("pages", 0))
            status = item.get("status", "pending")

            item_name = QTableWidgetItem(name)
            item_pages = QTableWidgetItem(pages)
            item_status = QTableWidgetItem(status)

            self.queue_table.setItem(row, 0, item_name)
            self.queue_table.setItem(row, 1, item_pages)
            self.queue_table.setItem(row, 2, item_status)

    def _on_start_clicked(self) -> None:

        self.worker_client.send_command("start")

    def _on_pause_clicked(self) -> None:
        if self.btn_pause.text() == "Resume":
            self.worker_client.send_command("resume")
        else:
            self.worker_client.send_command("pause")

    def _on_stop_clicked(self) -> None:
        self.worker_client.send_command("stop")

    def _on_open_folder_clicked(self) -> None:
        in_dir = Path(__file__).parent.parent.parent / "in"
        in_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(in_dir))
        else:
            subprocess.Popen(["xdg-open", str(in_dir)])
