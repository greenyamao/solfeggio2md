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
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

        # Graphics Canvas
        self.canvas_view = StaffGraphicsView(monitor_container)

        # Monitor Header Card
        mon_header = self._create_monitor_header()
        mon_layout.addWidget(mon_header)
        mon_layout.addWidget(self.canvas_view, stretch=1)


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
        self.btn_start = PrimaryPushButton(FluentIcon.PLAY, "Запустить конвейер", bar)
        self.btn_start.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.btn_start)

        self.btn_pause = PushButton(FluentIcon.PAUSE, "Пауза", bar)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        layout.addWidget(self.btn_pause)

        self.btn_stop = PushButton(FluentIcon.POWER_BUTTON, "Остановить", bar)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        layout.addWidget(self.btn_stop)

        self.btn_folder = PushButton(FluentIcon.FOLDER, "Папка in/", bar)
        self.btn_folder.clicked.connect(self._on_open_folder_clicked)
        layout.addWidget(self.btn_folder)

        layout.addStretch()

        # Status text in command bar
        self.lbl_bar_status = QLabel("Готов к работе", bar)
        self.lbl_bar_status.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 12px;")
        layout.addWidget(self.lbl_bar_status)

        return bar

    def _create_book_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        caption = CaptionLabel("ТЕКУЩАЯ КНИГА", card)
        caption.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
        header_layout.addWidget(caption)
        header_layout.addStretch()

        self.lbl_book_pages = CaptionLabel("0 / 0 стр.", card)
        self.lbl_book_pages.setStyleSheet("color: #38bdf8; font-weight: 600;")
        header_layout.addWidget(self.lbl_book_pages)
        layout.addLayout(header_layout)

        self.lbl_book_title = QLabel("Очередь ожидает команды", card)
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

        caption = CaptionLabel("ОЧЕРЕДЬ ФАЙЛОВ (in/)", card)
        caption.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 700;")
        layout.addWidget(caption)

        self.queue_table = TableWidget(card)
        self.queue_table.setColumnCount(3)
        self.queue_table.setHorizontalHeaderLabels(["Файл", "Стр.", "Статус"])
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

        self.lbl_mon_title = CaptionLabel("ЖИВОЙ МОНИТОР ТЕКУЩЕГО ШАГА", card)
        self.lbl_mon_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 700;")
        layout.addWidget(self.lbl_mon_title)

        layout.addStretch()

        self.badge_stage = QLabel("ФАЗА: ОЖИДАНИЕ", card)
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
        self.btn_fit.setToolTip("Вписать в окно")
        self.btn_fit.clicked.connect(self.canvas_view.fit_to_view)
        layout.addWidget(self.btn_fit)

        return card

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
                if title and self.lbl_mon_title.text() != f"ЖИВОЙ МОНИТОР: {title}":
                    self.lbl_mon_title.setText(f"ЖИВОЙ МОНИТОР: {title}")

                if path and path != self._current_image_path and os.path.isfile(path):
                    self._current_image_path = path
                    self.canvas_view.load_file(path)

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
                if self.lbl_bar_status.text() != "Пауза":
                    self.lbl_bar_status.setText("Пауза")
                    self.lbl_bar_status.setStyleSheet("color: #fb923c; font-weight: 600;")
                if self.btn_pause.text() != "Продолжить":
                    self.btn_pause.setText("Продолжить")
            else:
                if self.lbl_bar_status.text() != "Выполняется...":
                    self.lbl_bar_status.setText("Выполняется...")
                    self.lbl_bar_status.setStyleSheet("color: #4ade80; font-weight: 600;")
                if self.btn_pause.text() != "Пауза":
                    self.btn_pause.setText("Пауза")
        else:
            if self.lbl_bar_status.text() != "Остановлен / Готов":
                self.lbl_bar_status.setText("Остановлен / Готов")
                self.lbl_bar_status.setStyleSheet("color: #38bdf8; font-weight: 600;")

        # Active book
        book_name = metrics.get("current_book_name", "")
        if book_name and self.lbl_book_title.text() != book_name:
            self.lbl_book_title.setText(book_name)
        phase_name = metrics.get("current_phase_name", "Ожидание")
        phase_str = f"ФАЗА: {phase_name.upper()}"
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

        # Image from metrics if not already loaded
        img_path = metrics.get("current_page_image_path", "")
        if img_path and img_path != self._current_image_path and os.path.isfile(img_path):
            self._current_image_path = img_path
            self.canvas_view.load_file(img_path)

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
        if self.btn_pause.text() == "Продолжить":
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
