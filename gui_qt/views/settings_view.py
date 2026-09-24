"""
Settings & VLM Model Management View.

Allows configuring:
- VLM Backend (Embedded headless llama-server vs External LM Studio)
- Automated Hugging Face model download with live progress bar and speed
- Generation context size, image dimension capping, and OMR parameters
- Section filtering options
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
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
    LineEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    SpinBox,
    SubtitleLabel,
    SwitchButton,
)

from gui_qt.ipc.worker_process import MLPipelineProcessClient
from core.model_manager import ModelManager


class SettingsView(QWidget):
    """Settings and Model Download Manager."""

    def __init__(
        self,
        worker_client: MLPipelineProcessClient,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.worker_client = worker_client
        self.model_manager = ModelManager()

        self.root_dir = Path(__file__).parent.parent.parent.resolve()
        self.config_path = self.root_dir / "config.json"
        self.config = self._load_config()

        self._init_ui()
        self._setup_timer()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.is_file():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _init_ui(self) -> None:
        self.setObjectName("SettingsView")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        title = SubtitleLabel("ПАРАМЕТРЫ И МОДЕЛИ НЕЙРОСЕТИ", self)
        title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 700;")
        main_layout.addWidget(title)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget(scroll)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 8, 0)
        c_layout.setSpacing(12)

        # 1. VLM Backend & Model Card
        c_layout.addWidget(self._create_vlm_card())

        # 2. Performance & Tuning Card
        c_layout.addWidget(self._create_perf_card())

        # 3. Save Button
        save_bar = CardWidget(container)
        sb_layout = QHBoxLayout(save_bar)
        sb_layout.setContentsMargins(12, 10, 12, 10)
        sb_layout.addStretch()

        self.btn_save = PrimaryPushButton(FluentIcon.SAVE, "Сохранить параметры", save_bar)
        self.btn_save.clicked.connect(self._save_settings)
        sb_layout.addWidget(self.btn_save)

        c_layout.addWidget(save_bar)
        c_layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll, stretch=1)

    def _create_vlm_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        head = CaptionLabel("ДВИЖОК НЕЙРОСЕТИ VLM (VISION-LANGUAGE MODEL)", card)
        head.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        layout.addWidget(head)

        form = QFormLayout()
        form.setSpacing(10)

        # Backend Selector
        self.combo_backend = ComboBox(card)
        self.combo_backend.addItem("Встроенный llama-server (Рекомендуется, чисто кодом)")
        self.combo_backend.addItem("Внешний сервер LM Studio")
        cur_backend = self.config.get("vlm_backend", "embedded")
        self.combo_backend.setCurrentIndex(0 if cur_backend == "embedded" else 1)
        form.addRow("Режим инференса:", self.combo_backend)

        # Model Repo
        self.edit_repo = LineEdit(card)
        self.edit_repo.setText(self.config.get("vlm_model_repo", "lmstudio-community/Qwen3.5-9B-GGUF"))
        form.addRow("Hugging Face Репозиторий:", self.edit_repo)

        # GGUF Model File
        self.edit_model = LineEdit(card)
        self.edit_model.setText(self.config.get("vlm_model_file", "Qwen3.5-9B-Q4_K_M.gguf"))
        form.addRow("Имя файла весов (.gguf):", self.edit_model)

        # mmproj File
        self.edit_mmproj = LineEdit(card)
        self.edit_mmproj.setText(self.config.get("vlm_mmproj_file", "mmproj-Qwen3.5-9B-BF16.gguf"))
        form.addRow("Мультимодальный проектор:", self.edit_mmproj)

        # Port
        self.spin_port = SpinBox(card)
        self.spin_port.setRange(1000, 65535)
        self.spin_port.setValue(int(self.config.get("vlm_embedded_port", 1234)))
        form.addRow("Порт локального сервера:", self.spin_port)

        layout.addLayout(form)

        # Status and Downloader Section
        sep = QFrame(card)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1e293b;")
        layout.addWidget(sep)

        dl_header = QHBoxLayout()
        self.lbl_model_status = QLabel("Проверка файлов модели...", card)
        self.lbl_model_status.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 12px;")
        dl_header.addWidget(self.lbl_model_status)
        dl_header.addStretch()

        self.btn_download = PrimaryPushButton(FluentIcon.DOWNLOAD, "Скачать с Hugging Face", card)
        self.btn_download.clicked.connect(self._on_download_clicked)
        dl_header.addWidget(self.btn_download)

        self.btn_cancel_dl = PushButton("Отмена", card)
        self.btn_cancel_dl.setEnabled(False)
        self.btn_cancel_dl.clicked.connect(self._on_cancel_download_clicked)
        dl_header.addWidget(self.btn_cancel_dl)

        layout.addLayout(dl_header)

        # Progress bar
        self.dl_progress_bar = ProgressBar(card)
        self.dl_progress_bar.setValue(0)
        self.dl_progress_bar.setFixedHeight(8)
        layout.addWidget(self.dl_progress_bar)

        self.lbl_dl_detail = CaptionLabel("Готов к скачиванию", card)
        self.lbl_dl_detail.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_dl_detail)

        return card

    def _create_perf_card(self) -> CardWidget:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        head = CaptionLabel("ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ И ОБРАБОТКИ", card)
        head.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        layout.addWidget(head)

        form = QFormLayout()
        form.setSpacing(10)

        self.spin_context = SpinBox(card)
        self.spin_context.setRange(2048, 32768)
        self.spin_context.setSingleStep(512)
        self.spin_context.setValue(int(self.config.get("qwen_context_length", 4096)))
        form.addRow("Контекст токенов (4096):", self.spin_context)

        self.spin_max_dim = SpinBox(card)
        self.spin_max_dim.setRange(1024, 2560)
        self.spin_max_dim.setSingleStep(100)
        self.spin_max_dim.setValue(int(self.config.get("vlm_max_dim", 1600)))
        form.addRow("Лимит размера кадра VLM (1600 px):", self.spin_max_dim)

        self.spin_dpi = SpinBox(card)
        self.spin_dpi.setRange(100, 400)
        self.spin_dpi.setSingleStep(25)
        self.spin_dpi.setValue(int(self.config.get("dpi", 200)))
        form.addRow("DPI рендеринга PDF (200):", self.spin_dpi)

        self.spin_smt_tokens = SpinBox(card)
        self.spin_smt_tokens.setRange(128, 1024)
        self.spin_smt_tokens.setSingleStep(64)
        self.spin_smt_tokens.setValue(int(self.config.get("smt_max_tokens", 512)))
        form.addRow("Лимит токенов OMR (512):", self.spin_smt_tokens)

        # Switches
        self.sw_skip_vlm = SwitchButton("Пропустить VLM (только ноты и OMR)", card)
        self.sw_skip_vlm.setChecked(bool(self.config.get("skip_vlm", False)))
        form.addRow("Быстрый OMR режим:", self.sw_skip_vlm)

        self.sw_skip_front = SwitchButton("Пропускать оглавление/вводную часть", card)
        self.sw_skip_front.setChecked(bool(self.config.get("skip_front_matter", True)))
        form.addRow("Фильтр разделов:", self.sw_skip_front)

        layout.addLayout(form)
        return card

    def _setup_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._check_status)
        self._timer.start()

    def _check_status(self) -> None:
        # Check model file status
        st = self.model_manager.get_status(
            repo=self.edit_repo.text().strip(),
            model_file=self.edit_model.text().strip(),
            mmproj_file=self.edit_mmproj.text().strip(),
        )


        dl_state = self.model_manager.get_download_state()
        status_str = dl_state.get("status", "idle")

        if status_str == "downloading":
            self.btn_download.setEnabled(False)
            self.btn_cancel_dl.setEnabled(True)
            pct = float(dl_state.get("percent", 0.0))
            spd = float(dl_state.get("speed_mb_s", 0.0))
            eta = float(dl_state.get("eta_seconds", 0.0))
            cur_f = dl_state.get("current_file", "")
            f_idx = dl_state.get("file_index", 1)
            tot_f = dl_state.get("total_files", 2)

            self.dl_progress_bar.setValue(int(pct))
            self.lbl_model_status.setText(f"Скачивание файла {f_idx}/{tot_f}: {cur_f}")
            self.lbl_model_status.setStyleSheet("color: #38bdf8; font-weight: 600;")
            self.lbl_dl_detail.setText(
                f"Прогресс: {pct:.1f}% | Скорость: {spd:.1f} МБ/с | Осталось: ~{int(eta)} с"
            )
        else:
            self.btn_cancel_dl.setEnabled(False)
            self.btn_download.setEnabled(True)
            if st.get("ready", False):
                self.lbl_model_status.setText("Файлы модели найдены и готовы к работе")
                self.lbl_model_status.setStyleSheet("color: #4ade80; font-weight: 700;")
                self.dl_progress_bar.setValue(100)
                self.lbl_dl_detail.setText(f"Модель: {st.get('model_path', '')}")
            else:
                missing = st.get("missing", [])
                self.lbl_model_status.setText(f"Файлы не найдены ({len(missing)} шт.)")
                self.lbl_model_status.setStyleSheet("color: #fb923c; font-weight: 600;")
                self.lbl_dl_detail.setText("Нажмите 'Скачать с Hugging Face' для автозагрузки весов")

    def _on_download_clicked(self) -> None:
        self.worker_client.send_command(
            "download_model",
            repo=self.edit_repo.text().strip(),
            model_file=self.edit_model.text().strip(),
            mmproj_file=self.edit_mmproj.text().strip(),
        )

    def _on_cancel_download_clicked(self) -> None:
        self.worker_client.send_command("cancel_download")

    def _save_settings(self) -> None:
        backend_val = "embedded" if self.combo_backend.currentIndex() == 0 else "lm_studio"
        updates = {
            "vlm_backend": backend_val,
            "vlm_model_repo": self.edit_repo.text().strip(),
            "vlm_model_file": self.edit_model.text().strip(),
            "vlm_mmproj_file": self.edit_mmproj.text().strip(),
            "vlm_embedded_port": self.spin_port.value(),
            "qwen_context_length": self.spin_context.value(),
            "vlm_max_dim": self.spin_max_dim.value(),
            "dpi": self.spin_dpi.value(),
            "smt_max_tokens": self.spin_smt_tokens.value(),
            "skip_vlm": self.sw_skip_vlm.isChecked(),
            "skip_front_matter": self.sw_skip_front.isChecked(),
        }

        self.config.update(updates)
        self.config_path.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self.worker_client.send_command("update_config", config=updates)

        InfoBar.success(
            title="Параметры сохранены",
            content="Конфигурация успешно обновлена и синхронизирована с конвейером.",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3000,
            parent=self,
        )
