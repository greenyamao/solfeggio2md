"""
Fault-Tolerant Multi-File Batch Pipeline Runner.
Sequential Batch Passes: Slicing -> OMR -> VRAM Purge Barrier -> VLM (LM Studio) -> Markdown Assembly.
Resumable via atomic checkpoints (checkpoint.json) across 20+ books.
"""

import collections
import gc
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf as fitz
import torch

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.lmstudio_client import LMStudioClient
from core.page_preprocessor import PagePreprocessor, deskew_page, normalize_staff_crop, detect_and_split_spread
from core.layout_detector import LayoutDetector
from core.book_section_filter import BookSectionFilter
from core.windows_perf import enable_windows_high_performance
from core.process_telemetry import get_process_telemetry
from core.abc_bridge import ABCBridge
from core.model_manager import ModelManager
from core.embedded_llama_runner import EmbeddedLlamaRunner


DEFAULT_CONFIG_FILE = ROOT_DIR / "config.json"
QUEUE_STATE_FILE = ROOT_DIR / "queue_state.json"
DEFAULT_CONFIG = {
    "lm_host": "127.0.0.1",
    "lm_port": "1234",
    "lm_model": "qwen3.5-9b",
    "vlm_backend": "embedded",
    "vlm_model_repo": "lmstudio-community/Qwen3.5-9B-GGUF",
    "vlm_model_file": "Qwen3.5-9B-Q4_K_M.gguf",
    "vlm_mmproj_file": "mmproj-Qwen3.5-9B-BF16.gguf",
    "vlm_embedded_port": 1234,
    "lm_temperature": 0.1,
    "lm_max_tokens": 2048,
    "output_dir": "output",
    "dpi": 200,
    "delay": 0.2,
    "smt_max_tokens": 512,
    "qwen_context_length": 4096,
    "vlm_max_dim": 1600,
    "qwen_parallel": 1,
    "qwen_eval_batch_size": 2048,
    "qwen_flash_attention": True,
    "qwen_offload_kv_cache_to_gpu": True,
    "smt_model": "antoniorv6/smt-grandstaff",
    "overwrite": False,
    "smt_device": "cuda" if torch.cuda.is_available() else "cpu",
    "enable_score_enhancer": True,
    "enable_cugan_sr": True,
    "skip_vlm": True,
    "skip_front_matter": True,
    "skip_back_matter": True,
    "system_prompt": (
        "Ты — строгий OCR-транскрибатор. Перенеси весь печатный текст страницы в чистый Markdown дословно.\n"
        "Сохраняй иерархию заголовков (#, ##, ###), таблицы и списки.\n"
        "ВАЖНО: Если на странице встречаются технические метки вида <!-- MUSIC_STUB_ID:... -->, "
        "ОБЯЗАТЕЛЬНО оставь их в тексте на тех же местах без малейших изменений. Ничего не додумывай от себя."
    ),
}


class PipelineBatchRunner:
    """
    Manages queue of PDF documents and executes sequential batch passes
    with full checkpointing and fault tolerance.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_FILE
        self.config = self.load_config()
        cfg_out = self.config.get("output_dir", "output")
        try:
            p_out = Path(cfg_out)
            if not p_out.is_absolute():
                target_out = (self.config_path.parent / p_out).resolve()
            else:
                target_out = p_out.resolve()
            target_out.mkdir(parents=True, exist_ok=True)
            self.output_root = target_out
        except Exception:
            fallback = (self.config_path.parent / "output").resolve()
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                self.output_root = fallback
            except Exception:
                self.output_root = (ROOT_DIR / "output").resolve()
                self.output_root.mkdir(parents=True, exist_ok=True)

        cfg_in = self.config.get("input_dir", "in")
        try:
            p_in = Path(cfg_in)
            if not p_in.is_absolute():
                target_in = (self.config_path.parent / p_in).resolve()
            else:
                target_in = p_in.resolve()
            target_in.mkdir(parents=True, exist_ok=True)
            self.input_dir = target_in
        except Exception:
            fallback_in = (self.config_path.parent / "in").resolve()
            try:
                fallback_in.mkdir(parents=True, exist_ok=True)
                self.input_dir = fallback_in
            except Exception:
                self.input_dir = (ROOT_DIR / "in").resolve()
                self.input_dir.mkdir(parents=True, exist_ok=True)

        # Queue of items: [{"id": "1", "name": "1.pdf", "path": "in/1.pdf", "pages": 48, "status": "pending"}]
        self.queue: List[Dict[str, Any]] = []

        # State tracking
        self.is_running = False
        self.is_paused = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # set means NOT paused

        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time: float = 0.0

        # Real-time HUD Metrics
        self._log_history: collections.deque = collections.deque(maxlen=300)
        self.metrics: Dict[str, Any] = {
            "is_running": False,
            "is_paused": False,
            "queue_progress_pct": 0.0,
            "book_progress_pct": 0.0,
            "current_book_index": 0,
            "total_books": 0,
            "current_book_name": "",
            "current_phase_name": "Готов к запуску",
            "current_item_detail": "Очередь ожидает команды",
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": 0.0,
            "vram_allocated_mb": 0.0,
            "last_log": "Инициализация выполнена",
            "logs": [],
            "vlm_tok_per_sec": 0.0,
            "last_page_seconds": 0.0,
            "last_page_tokens": 0,
            "current_page_image_url": "",
            "current_page_image_path": "",
            "phase_progress": {
                "phase1": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                "phase2": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                "phase3": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                "phase4": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
            },
        }

        # Lazy engines
        self.layout_detector: Optional[LayoutDetector] = None
        self.omr_engine = None
        self.model_manager = ModelManager()
        self.embedded_runner = EmbeddedLlamaRunner(
            port=int(self.config.get("vlm_embedded_port", 1234)),
            log_callback=lambda tag, msg: self.log_event(tag, msg),
        )
        self.lm_client = LMStudioClient(
            host=self.config.get("lm_host", "127.0.0.1"),
            port=self.config.get("lm_port", "1234"),
        )
        self.bridge = ABCBridge()
        self.on_log_event: Optional[Any] = None
        self.on_frame_update: Optional[Any] = None

        # Log initial event
        self.log_event("SYSTEM", "Система инициализирована. Готова к обработке очереди.")

        # Initial queue population and directory sync
        self._load_existing_queue()

    def log_event(self, tag: str, msg: str, level: str = "INFO") -> None:
        """Appends a structured event to the real-time circular log buffer."""
        now_str = datetime.now().strftime("%H:%M:%S")
        entry = {
            "time": now_str,
            "tag": tag,
            "msg": msg,
            "level": level,
        }
        with self._lock:
            self._log_history.append(entry)
            self.metrics["last_log"] = f"[{tag}] {msg}"
        if self.on_log_event is not None:
            try:
                self.on_log_event(entry)
            except Exception:
                pass


    def load_config(self) -> Dict[str, Any]:
        cfg = dict(DEFAULT_CONFIG)
        if self.config_path.is_file():
            try:
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                cfg.update(loaded)
            except Exception:
                pass
        return cfg

    def save_config(self, new_cfg: Dict[str, Any]) -> None:
        with self._lock:
            self.config.update(new_cfg)
            self.config_path.write_text(
                json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            cfg_out = self.config.get("output_dir", "output")
            try:
                p_out = Path(cfg_out)
                if not p_out.is_absolute():
                    target_out = (self.config_path.parent / p_out).resolve()
                else:
                    target_out = p_out.resolve()
                target_out.mkdir(parents=True, exist_ok=True)
                self.output_root = target_out
            except Exception:
                fallback = (self.config_path.parent / "output").resolve()
                try:
                    fallback.mkdir(parents=True, exist_ok=True)
                    self.output_root = fallback
                except Exception:
                    self.output_root = (ROOT_DIR / "output").resolve()
                    self.output_root.mkdir(parents=True, exist_ok=True)

            cfg_in = self.config.get("input_dir", "in")
            try:
                p_in = Path(cfg_in)
                if not p_in.is_absolute():
                    target_in = (self.config_path.parent / p_in).resolve()
                else:
                    target_in = p_in.resolve()
                target_in.mkdir(parents=True, exist_ok=True)
                self.input_dir = target_in
            except Exception:
                fallback_in = (self.config_path.parent / "in").resolve()
                try:
                    fallback_in.mkdir(parents=True, exist_ok=True)
                    self.input_dir = fallback_in
                except Exception:
                    self.input_dir = (ROOT_DIR / "in").resolve()
                    self.input_dir.mkdir(parents=True, exist_ok=True)
            self.lm_client.host = self.config.get("lm_host", "127.0.0.1")
            self.lm_client.port = str(self.config.get("lm_port", "1234"))

    def _save_queue_state(self) -> None:
        try:
            # Save paths as relative (in/filename.pdf) so the folder is fully portable
            clean_queue = []
            for q in self.queue:
                q_copy = dict(q)
                fname = Path(q_copy.get("path", "")).name
                q_copy["path"] = f"in/{fname}"
                clean_queue.append(q_copy)
            data = {
                "queue": clean_queue,
                "last_saved": datetime.now().isoformat()
            }
            QUEUE_STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _sync_with_in_dir(self) -> None:
        """Dynamically synchronizes queue and queue_state.json with physical files in `in/` directory."""
        if not self.input_dir.is_dir():
            self.input_dir.mkdir(parents=True, exist_ok=True)

        disk_files = {p.name: p for p in sorted(self.input_dir.glob("*.pdf"))}
        changed = False

        # 1. Prune items from queue if their physical file no longer exists in `in/` (only for pending or error items)
        new_queue = []
        for q in self.queue:
            fname = Path(q.get("path", "")).name
            if fname not in disk_files and q.get("status") in ("pending", "error"):
                changed = True
                continue
            new_queue.append(q)
        self.queue = new_queue

        # 2. Add any newly appeared PDF files in `in/`
        existing_names = {Path(q.get("path", "")).name for q in self.queue}
        for fname, pdf_path in disk_files.items():
            if fname not in existing_names:
                self._enqueue_file(str(pdf_path))
                changed = True

        if changed:
            for i, q in enumerate(self.queue, 1):
                q["id"] = str(i)
            self.metrics["total_books"] = len(self.queue)
            self._save_queue_state()

    def _load_existing_queue(self) -> None:
        """Loads queue from queue_state.json and synchronizes with `in/` folder."""
        if QUEUE_STATE_FILE.is_file():
            try:
                state = json.loads(QUEUE_STATE_FILE.read_text(encoding="utf-8"))
                for item in state.get("queue", []):
                    raw_p = item.get("path", "")
                    p = Path(raw_p)
                    if not p.is_absolute():
                        p = (ROOT_DIR / p).resolve()
                    if not p.is_file():
                        p_fallback = (self.input_dir / Path(raw_p).name).resolve()
                        if p_fallback.is_file():
                            p = p_fallback
                    if p.is_file():
                        item["path"] = f"in/{p.name}"
                        book_title = p.stem
                        chk = self._read_checkpoint(book_title)
                        if chk:
                            item["status"] = chk.get("status", item.get("status", "pending"))
                        self.queue.append(item)
            except Exception:
                pass

        self._sync_with_in_dir()

    def _enqueue_file(self, path_str: str) -> None:
        p = Path(path_str).resolve()
        if not p.is_file() or p.suffix.lower() != ".pdf":
            return
        if any(Path(item.get("path", "")).name == p.name for item in self.queue):
            return

        pages_count = 0
        if p.stat().st_size >= 100:
            try:
                with fitz.open(p) as doc:
                    pages_count = len(doc)
            except Exception:
                pages_count = 0

        # Check existing checkpoint without creating empty directories
        book_title = p.stem
        chk = self._read_checkpoint(book_title)
        status = "completed" if chk and chk.get("status") == "completed" else "pending"

        self.queue.append(
            {
                "id": str(len(self.queue) + 1),
                "name": p.name,
                "path": f"in/{p.name}",
                "pages": pages_count,
                "status": status,
                "progress_pct": 100.0 if status == "completed" else 0.0,
            }
        )

    def add_pdf(self, file_path: str) -> bool:
        with self._lock:
            p = Path(file_path).resolve()
            if not p.is_file() or p.suffix.lower() != ".pdf":
                return False
            self._enqueue_file(str(p))
            for i, q in enumerate(self.queue, 1):
                q["id"] = str(i)
            self.metrics["total_books"] = len(self.queue)
            self._save_queue_state()
            return True

    def remove_pdf(self, item_id: str) -> bool:
        with self._lock:
            idx = next((i for i, item in enumerate(self.queue) if item["id"] == item_id), None)
            if idx is not None:
                self.queue.pop(idx)
                for i, q in enumerate(self.queue, 1):
                    q["id"] = str(i)
                self.metrics["total_books"] = len(self.queue)
                self._save_queue_state()
                return True
            return False

    def clear_queue(self) -> None:
        with self._lock:
            if not self.is_running:
                self.queue.clear()
            self.metrics["total_books"] = len(self.queue)
            self._save_queue_state()

    def get_queue(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._sync_with_in_dir()
            return [dict(q) for q in self.queue]

    def scan_inputs(self) -> List[Dict[str, Any]]:
        """Scans the in/ folder and refreshes the file queue."""
        return self.get_queue()

    def get_queue_summary(self) -> Dict[str, Any]:
        """Returns consolidated queue and status dictionary."""
        q = self.get_queue()
        m = self.get_metrics()
        return {"queue": q, "metrics": m}


    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            m = dict(self.metrics)
            m["total_books"] = len(self.queue)
            m["logs"] = list(self._log_history)
            if self.is_running and self._start_time > 0:
                m["elapsed_seconds"] = round(time.time() - self._start_time, 1)

        # Merge process-isolated telemetry (CPU, RAM RSS, VRAM, GPU compute)
        try:
            telemetry = get_process_telemetry()
            m["process_telemetry"] = telemetry
            m["cpu_percent"] = telemetry.get("cpu_percent", 0.0)
            m["ram_rss_mb"] = telemetry.get("ram_rss_mb", 0.0)
            m["vram_allocated_mb"] = telemetry.get("vram_allocated_mb", 0.0)
            m["vram_reserved_mb"] = telemetry.get("vram_reserved_mb", 0.0)
            m["gpu_compute_percent"] = telemetry.get("gpu_compute_percent", 0.0)
            m["device_name"] = telemetry.get("device_name", "CPU")
            m["runtime_mode"] = telemetry.get("runtime_mode", "CPU Mode")
        except Exception:
            pass

        return m

    # ---------------- Checkpointing ---------------- #

    def _get_book_dir(self, book_title: str) -> Path:
        d = self.output_root / book_title
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _read_checkpoint(self, book_title: str) -> Optional[Dict[str, Any]]:
        chk_file = self.output_root / book_title / "checkpoint.json"
        if chk_file.is_file():
            try:
                return json.loads(chk_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _write_checkpoint(self, book_title: str, data: Dict[str, Any]) -> None:
        with self._lock:
            chk_file = self._get_book_dir(book_title) / "checkpoint.json"
            data["last_updated"] = datetime.now().isoformat()
            chk_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Control Methods ---------------- #

    def start(self) -> bool:
        with self._lock:
            if self.is_running:
                if self.is_paused:
                    self.resume()
                    return True
                return False

            self.is_running = True
            self.is_paused = False
            self._stop_event.clear()
            self._pause_event.set()

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def pause(self) -> None:
        with self._lock:
            if self.is_running and not self.is_paused:
                self.is_paused = True
                self._pause_event.clear()
                self.metrics["is_paused"] = True
                self.metrics["current_phase_name"] = "Приостановлено пользователем"

    def resume(self) -> None:
        with self._lock:
            if self.is_running and self.is_paused:
                self.is_paused = False
                self._pause_event.set()
                self.metrics["is_paused"] = False

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused
        if hasattr(self, "embedded_runner"):
            try:
                self.embedded_runner.stop()
            except Exception:
                pass
        if hasattr(self, "lm_client") and getattr(self.lm_client, "was_loaded_by_client", False):
            try:
                self.lm_client.unload_model()
            except Exception:
                pass
        with self._lock:
            self.is_running = False
            self.is_paused = False
            self.metrics["is_running"] = False
            self.metrics["is_paused"] = False
            self.metrics["current_phase_name"] = "Остановлено пользователем"

    # ---------------- Orchestration Loop ---------------- #

    def _run_loop(self) -> None:
        with self._lock:
            self._start_time = time.time()
        start_time = self._start_time
        active_items = [q for q in self.queue if q["status"] != "completed"]
        total_books = len(self.queue)

        try:
            for book_idx, item in enumerate(self.queue, start=1):
                if self._stop_event.is_set():
                    break

                self._check_pause()
                if item["status"] == "completed" and not self.config.get("overwrite", False):
                    continue

                pdf_path = Path(item["path"])
                if not pdf_path.is_absolute():
                    pdf_path = (ROOT_DIR / pdf_path).resolve()
                book_title = pdf_path.stem
                item["status"] = "processing"

                with self._lock:
                    self.metrics.update(
                        {
                            "is_running": True,
                            "current_book_index": book_idx,
                            "total_books": total_books,
                            "current_book_name": item["name"],
                            "book_progress_pct": 0.0,
                            "queue_progress_pct": round(((book_idx - 1) / max(1, total_books)) * 100, 1),
                            "phase_progress": {
                                "phase1": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                                "phase2": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                                "phase3": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                                "phase4": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": "Ожидание"},
                            },
                        }
                    )

                # Process single book through the 4-Phase pipeline
                success = self._process_book(pdf_path, item)
                if not success and self._stop_event.is_set():
                    break

                item["status"] = "completed" if success else "error"
                item["progress_pct"] = 100.0 if success else item.get("progress_pct", 0.0)

            with self._lock:
                self.metrics["queue_progress_pct"] = 100.0
                self.metrics["book_progress_pct"] = 100.0
                self.metrics["current_phase_name"] = "Очередь полностью обработана"
                self.metrics["current_item_detail"] = "Все задачи завершены"
                for p_key in ("phase1", "phase2", "phase3", "phase4"):
                    if p_key in self.metrics.get("phase_progress", {}):
                        self.metrics["phase_progress"][p_key]["pct"] = 100.0
                        self.metrics["phase_progress"][p_key]["status"] = "completed"

        except Exception as e:
            with self._lock:
                self.metrics["current_phase_name"] = "Ошибка в конвейере"
                self.metrics["last_log"] = f"Исключение: {str(e)}"
        finally:
            self._purge_vram()
            if hasattr(self, "embedded_runner"):
                try:
                    self.embedded_runner.stop()
                except Exception:
                    pass
            if hasattr(self, "lm_client") and getattr(self.lm_client, "was_loaded_by_client", False):
                try:
                    self.lm_client.unload_model()
                except Exception:
                    pass
            with self._lock:
                self.is_running = False
                self.is_paused = False
                self.metrics["is_running"] = False
                self.metrics["is_paused"] = False
                self.metrics["elapsed_seconds"] = round(time.time() - start_time, 1)

    def _check_pause(self) -> None:
        while not self._pause_event.is_set() and not self._stop_event.is_set():
            self._pause_event.wait(timeout=0.2)

    # ---------------- 4-Phase Book Processing ---------------- #

    def _process_book(self, pdf_path: Path, item: Dict[str, Any]) -> bool:
        enable_windows_high_performance()
        book_title = pdf_path.stem
        book_dir = self._get_book_dir(book_title)

        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        for d in (crops_dir, masked_dir, raw_md_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

        overwrite = self.config.get("overwrite", False)
        chk = self._read_checkpoint(book_title) or {
            "book_title": book_title,
            "status": "in_progress",
            "phases": {
                "slicing": {"completed": False, "pages_done": 0, "total_pages": 0},
                "omr": {"completed": False, "crops_done": 0, "total_crops": 0},
                "vlm": {"completed": False, "pages_done": 0, "total_pages": 0},
                "assembly": {"completed": False},
            },
        }

        # Physical disk inspection: sync checkpoint with actual files on disk
        if not overwrite:
            tot_p = chk["phases"]["slicing"].get("total_pages", 0)
            if tot_p == 0:
                try:
                    with fitz.open(pdf_path) as doc:
                        tot_p = sum(
                            2 if (1.22 <= (p.rect.width / float(max(1.0, p.rect.height))) <= 2.2 and p.rect.height >= 300) else 1
                            for p in doc
                        )
                except Exception:
                    tot_p = 0

            existing_m = len([f for f in masked_dir.glob("page_*_masked.png") if f.stat().st_size > 1000])
            if existing_m >= tot_p and tot_p > 0:
                chk["phases"]["slicing"]["completed"] = True
                chk["phases"]["slicing"]["pages_done"] = tot_p
                chk["phases"]["slicing"]["total_pages"] = tot_p

            c_files = [f for f in crops_dir.glob("*.png") if not f.name.endswith("_deskew.png")]
            existing_a = len([f for f in crops_dir.glob("*.abc") if f.stat().st_size > 0])
            if len(c_files) > 0 and existing_a >= len(c_files):
                chk["phases"]["omr"]["completed"] = True
                chk["phases"]["omr"]["crops_done"] = len(c_files)
                chk["phases"]["omr"]["total_crops"] = len(c_files)

            m_files = list(masked_dir.glob("page_*_masked.png"))
            existing_r = len([f for f in raw_md_dir.glob("page_*_raw.md") if f.stat().st_size > 0])
            if len(m_files) > 0 and existing_r >= len(m_files):
                chk["phases"]["vlm"]["completed"] = True
                chk["phases"]["vlm"]["pages_done"] = len(m_files)
                chk["phases"]["vlm"]["total_pages"] = len(m_files)

            self._write_checkpoint(book_title, chk)

        # ---------------- 4-Phase Progress Initialization ---------------- #
        p1_comp = bool(chk["phases"]["slicing"].get("completed", False) and not overwrite)
        p1_tot = int(chk["phases"]["slicing"].get("total_pages", 0))
        p1_done = p1_tot if p1_comp else int(chk["phases"]["slicing"].get("pages_done", 0))

        p2_comp = bool(chk["phases"]["omr"].get("completed", False) and not overwrite)
        p2_tot = int(chk["phases"]["omr"].get("total_crops", 0))
        p2_done = p2_tot if p2_comp else int(chk["phases"]["omr"].get("crops_done", 0))

        p3_comp = bool(chk["phases"]["vlm"].get("completed", False) and not overwrite)
        p3_tot = int(chk["phases"]["vlm"].get("total_pages", 0))
        p3_done = p3_tot if p3_comp else int(chk["phases"]["vlm"].get("pages_done", 0))

        p4_comp = bool(chk["phases"]["assembly"].get("completed", False) and not overwrite)

        self._update_phase_progress(
            "phase1",
            pct=100.0 if p1_comp else (round((p1_done / p1_tot) * 100.0, 1) if p1_tot > 0 else 0.0),
            done=p1_done,
            total=p1_tot,
            status="completed" if p1_comp else "pending",
            detail="Готово" if p1_comp else "Ожидание",
        )
        self._update_phase_progress(
            "phase2",
            pct=100.0 if p2_comp else (round((p2_done / p2_tot) * 100.0, 1) if p2_tot > 0 else 0.0),
            done=p2_done,
            total=p2_tot,
            status="completed" if p2_comp else "pending",
            detail="Готово" if p2_comp else "Ожидание",
        )
        self._update_phase_progress(
            "phase3",
            pct=100.0 if p3_comp else (round((p3_done / p3_tot) * 100.0, 1) if p3_tot > 0 else 0.0),
            done=p3_done,
            total=p3_tot,
            status="completed" if p3_comp else "pending",
            detail="Готово" if p3_comp else "Ожидание",
        )
        self._update_phase_progress(
            "phase4",
            pct=100.0 if p4_comp else 0.0,
            done=0,
            total=0,
            status="completed" if p4_comp else "pending",
            detail="Готово" if p4_comp else "Ожидание",
        )

        # ---------------- Phases 1 & 2: Pipelined Streaming (Slicing + OMR) ---------------- #
        slicing_done = chk["phases"]["slicing"].get("completed", False) and not overwrite
        omr_done = chk["phases"]["omr"].get("completed", False) and not overwrite

        if not (slicing_done and omr_done):
            if slicing_done:
                # Slicing already done on disk; only run remaining OMR
                self._update_hud("Фаза 2/4: Оптическое распознавание нот (Transcoda-59M)", 25.0)
                ok = self._phase_2_omr(crops_dir, chk, overwrite)
                if not ok:
                    return False
                chk["phases"]["omr"]["completed"] = True
                self._write_checkpoint(book_title, chk)
            else:
                # Pipelined concurrent execution of Slicing and OMR
                self._update_hud("Фазы 1-2: Потоковая нарезка и распознавание нот (YOLO + Transcoda)", 0.0)
                ok = self._phase_1_and_2_pipelined(pdf_path, crops_dir, masked_dir, chk, overwrite)
                if not ok:
                    return False
                chk["phases"]["slicing"]["completed"] = True
                chk["phases"]["omr"]["completed"] = True
                self._write_checkpoint(book_title, chk)

        # ---------------- VRAM Purge Barrier before VLM ---------------- #
        self._update_hud("Очистка VRAM GPU перед VLM...", 50.0)
        self._purge_vram()

        # ---------------- Phase 3: VLM Text Recognition ---------------- #
        if not chk["phases"]["vlm"]["completed"] or overwrite:
            self._update_hud("Фаза 3/4: Извлечение текста книги (LM Studio VLM)", 55.0)
            ok = self._phase_3_vlm(masked_dir, raw_md_dir, chk, overwrite)
            if not ok:
                return False
            chk["phases"]["vlm"]["completed"] = True
            self._write_checkpoint(book_title, chk)

        # ---------------- Phase 4: Final Assembly ---------------- #
        self._update_hud("Фаза 4/4: Сборка итогового Markdown издания", 90.0)
        ok = self._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)
        if not ok:
            return False

        chk["phases"]["assembly"]["completed"] = True
        chk["status"] = "completed"
        self._write_checkpoint(book_title, chk)
        self._update_hud(f"Книга {book_title} успешно завершена", 100.0)
        return True

    def _update_phase_progress(
        self,
        phase_id: str,
        pct: float,
        done: int = 0,
        total: int = 0,
        status: str = "running",
        detail: str = "",
    ) -> None:
        with self._lock:
            if "phase_progress" not in self.metrics:
                self.metrics["phase_progress"] = {
                    "phase1": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": ""},
                    "phase2": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": ""},
                    "phase3": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": ""},
                    "phase4": {"pct": 0.0, "done": 0, "total": 0, "status": "pending", "detail": ""},
                }
            p = self.metrics["phase_progress"].setdefault(phase_id, {})
            p["pct"] = round(min(100.0, max(0.0, float(pct))), 1)
            p["done"] = int(done)
            p["total"] = int(total)
            p["status"] = str(status)
            if detail:
                p["detail"] = str(detail)

            # Recalculate book_progress_pct across all 4 phases (weights: P1=25%, P2=25%, P3=35%, P4=15%)
            p1 = self.metrics["phase_progress"]["phase1"]["pct"]
            p2 = self.metrics["phase_progress"]["phase2"]["pct"]
            p3 = self.metrics["phase_progress"]["phase3"]["pct"]
            p4 = self.metrics["phase_progress"]["phase4"]["pct"]
            book_overall_pct = round((p1 * 0.25) + (p2 * 0.25) + (p3 * 0.35) + (p4 * 0.15), 1)
            self.metrics["book_progress_pct"] = book_overall_pct

            # Calculate continuous queue_progress_pct
            cur_idx = max(1, self.metrics.get("current_book_index", 1))
            tot_books = max(1, self.metrics.get("total_books", len(self.queue) or 1))
            continuous_queue_pct = round((((cur_idx - 1) + (book_overall_pct / 100.0)) / tot_books) * 100.0, 1)
            self.metrics["queue_progress_pct"] = min(100.0, max(0.0, continuous_queue_pct))

            if detail:
                self.metrics["current_item_detail"] = detail

    def _update_hud(self, phase_name: str, book_pct: float, detail: str = "") -> None:
        with self._lock:
            self.metrics["current_phase_name"] = phase_name
            self.metrics["book_progress_pct"] = round(book_pct, 1)
            if detail:
                self.metrics["current_item_detail"] = detail

    def _create_batch_composite(
        self, crops_bgr: List[np.ndarray], titles: List[str], classes: List[str]
    ) -> Optional[np.ndarray]:
        """Creates a unified vertical collage of all crops in the current OMR batch."""
        if not crops_bgr:
            return None

        max_w = max(c.shape[1] for c in crops_bgr)
        pad_x = 24
        banner_h = 24
        gap = 12

        total_h = sum(c.shape[0] + banner_h + gap for c in crops_bgr) + gap
        canvas = np.full((total_h, max_w + pad_x * 2, 3), (15, 23, 42), dtype=np.uint8)  # Deep slate #0f172a

        y = gap
        for i, (c, t, cls_name) in enumerate(zip(crops_bgr, titles, classes)):
            h, w = c.shape[:2]
            header_text = f"[{i+1}/{len(crops_bgr)}] {t} ({cls_name})"
            col = (250, 204, 21) if "grand" in cls_name else (56, 189, 248)  # Gold for grand, Cyan for staff
            cv2.putText(
                canvas,
                header_text,
                (pad_x, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                col,
                1,
                cv2.LINE_AA,
            )
            y += banner_h
            canvas[y : y + h, pad_x : pad_x + w] = c
            cv2.rectangle(canvas, (pad_x - 1, y - 1), (pad_x + w, y + h), (51, 65, 85), 1)
            y += h + gap

        return canvas

    def _transcribe_crops_batch_files(self, crops_dir: Path, batch_files: List[Path]) -> int:
        """Transcribes a batch of crop image files via self.omr_engine and persists .abc and .kern outputs."""
        if not batch_files or self.omr_engine is None:
            return 0

        batch_crops_bgr = []
        batch_classes = []
        batch_titles = []
        valid_batch_files = []

        for cf in batch_files:
            cls_name = "staff"
            if "grand_staff" in cf.stem:
                cls_name = "grand_staff"
            elif "system" in cf.stem:
                cls_name = "system"

            deskew_file = crops_dir / f"{cf.stem}_deskew.png"
            if deskew_file.is_file():
                crop_bgr = cv2.imread(str(deskew_file))
            else:
                raw_crop = cv2.imread(str(cf))
                if raw_crop is not None:
                    dewarped_bgr, _, _ = normalize_staff_crop(
                        raw_crop,
                        notation_class=cls_name,
                        enhance_sr=False
                    )
                    cv2.imwrite(str(deskew_file), dewarped_bgr)
                    crop_bgr = dewarped_bgr
                else:
                    crop_bgr = None

            if crop_bgr is None:
                continue

            batch_crops_bgr.append(crop_bgr)
            batch_classes.append(cls_name)
            batch_titles.append(cf.stem)
            valid_batch_files.append(cf)

        if not batch_crops_bgr:
            return 0

        # Generate and broadcast consolidated multi-crop visual batch preview
        composite_bgr = self._create_batch_composite(batch_crops_bgr, batch_titles, batch_classes)
        if composite_bgr is not None:
            preview_file = crops_dir.parent / "omr_batch_preview.png"
            try:
                cv2.imwrite(str(preview_file), composite_bgr)
                rel_prev = preview_file.relative_to(self.output_root).as_posix()
                with self._lock:
                    self.metrics["current_page_image_url"] = f"/output/{rel_prev}"
                    self.metrics["current_page_image_path"] = str(preview_file)
                if self.on_frame_update is not None:
                    self.on_frame_update(composite_bgr, {
                        "path": str(preview_file),
                        "stage": "phase2",
                        "title": f"OMR Пачка ({len(batch_crops_bgr)} станов)",
                    })
            except Exception:
                pass

        results = self.omr_engine.transcribe_crops_batch(
            crops_bgr=batch_crops_bgr,
            notation_classes=batch_classes,
            titles=batch_titles
        )


        saved_count = 0
        for cf, res in zip(valid_batch_files, results):
            abc_content = res.get("abc", "")
            kern_content = res.get("raw_kern", "")
            if abc_content:
                abc_file = crops_dir / f"{cf.stem}.abc"
                abc_file.write_text(abc_content, encoding="utf-8")
            if kern_content:
                kern_file = crops_dir / f"{cf.stem}.kern"
                kern_file.write_text(kern_content, encoding="utf-8")
            saved_count += 1

        return saved_count

    # ---------------- Streaming Pipelined Phase 1 & 2 ---------------- #

    def _phase_1_and_2_pipelined(
        self, pdf_path: Path, crops_dir: Path, masked_dir: Path, chk: Dict[str, Any], overwrite: bool
    ) -> bool:
        """
        True concurrent multi-threaded Producer-Consumer pipeline for Phase 1 (Slicing) and Phase 2 (OMR).
        Thread 1 (Slicing Producer): Continuously slices pages at full speed without stalling.
        Thread 2 (OMR Consumer): Concurrently transcribes newly buffered crops on GPU in mini-batches.
        Both models (YOLO OLA ~1GB + Transcoda ~1.1GB = ~2.1GB) fit comfortably in 8GB VRAM.
        """
        with fitz.open(pdf_path) as doc:
            total_sheets = len(doc)

            # Pre-compute sheet to book page mapping (two-page spreads produce 2 book pages)
            sheet_mapping: List[Tuple[int, ...]] = []
            cur_bp = 1
            for s_idx in range(total_sheets):
                p = doc[s_idx]
                w, h = p.rect.width, p.rect.height
                is_spread = (1.22 <= (w / float(max(1.0, h))) <= 2.2 and h >= 300)
                if is_spread:
                    sheet_mapping.append((cur_bp, cur_bp + 1))
                    cur_bp += 2
                else:
                    sheet_mapping.append((cur_bp,))
                    cur_bp += 1

            total_book_pages = cur_bp - 1
            chk["phases"]["slicing"]["total_pages"] = total_book_pages

            existing_valid_pages = set()
            if not overwrite:
                for f in masked_dir.glob("page_*_masked.png"):
                    m = re.search(r"page_(\d+)_masked", f.stem)
                    if m and f.stat().st_size > 1000:
                        existing_valid_pages.add(int(m.group(1)))

            # Lazy init engines (LayoutDetector & OMREngine)
            if self.layout_detector is None:
                weights_path = ROOT_DIR / "weights" / "ola-layout-analysis-2.0-2025-03-09.pt"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.layout_detector = LayoutDetector(weights_path=str(weights_path), device=device)

            if self.omr_engine is None:
                from core.omr_engine import OMREngine
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.omr_engine = OMREngine(
                    device=device,
                    enable_score_enhancer=self.config.get("enable_score_enhancer", True),
                    enable_cugan=self.config.get("enable_cugan_sr", True)
                )

            section_filter = BookSectionFilter(
                skip_front_matter=self.config.get("skip_front_matter", True),
                skip_back_matter=self.config.get("skip_back_matter", True),
            )
            sections = section_filter.analyze_document_sections(pdf_path)

            dpi = int(self.config.get("dpi", 200))
            pages_done_count = len(existing_valid_pages)

            existing_valid_abcs = set()
            if not overwrite:
                for f in crops_dir.glob("*.abc"):
                    if f.stat().st_size > 0:
                        existing_valid_abcs.add(f.stem)

            # Thread-safe queue of crop file paths waiting for OMR
            omr_queue: queue.Queue = queue.Queue()

            # Pre-enqueue any existing un-transcribed crops from disk if resuming
            if not overwrite:
                disk_crops = sorted([f for f in crops_dir.glob("*.png") if not f.name.endswith("_deskew.png")])
                for dc in disk_crops:
                    if dc.stem not in existing_valid_abcs:
                        omr_queue.put(dc)

            stats_lock = threading.Lock()
            stats = {
                "crops_cut": len(existing_valid_abcs) + omr_queue.qsize(),
                "crops_done": len(existing_valid_abcs),
            }

            chk["phases"]["omr"]["crops_done"] = stats["crops_done"]
            chk["phases"]["omr"]["total_crops"] = max(1, stats["crops_cut"])

            # Initial progress update
            p1_init_pct = round((pages_done_count / max(1, total_book_pages)) * 100.0, 1)
            p2_init_pct = round((stats["crops_done"] / max(1, stats["crops_cut"])) * 100.0, 1)
            self._update_phase_progress("phase1", p1_init_pct, pages_done_count, total_book_pages, "running", f"Стр. {pages_done_count}/{total_book_pages}")
            self._update_phase_progress("phase2", p2_init_pct, stats["crops_done"], stats["crops_cut"], "running", f"Станов: {stats['crops_done']}/{stats['crops_cut']}")

            producer_done = threading.Event()
            thread_errors: List[Exception] = []

            # ---------------- Producer: Slicing Worker ---------------- #
            def _producer_slicing() -> None:
                try:
                    nonlocal pages_done_count
                    for s_idx in range(total_sheets):
                        if self._stop_event.is_set() or thread_errors:
                            break
                        self._check_pause()

                        book_pages = sheet_mapping[s_idx]
                        if not overwrite and all(bp in existing_valid_pages for bp in book_pages):
                            continue

                        page = doc[s_idx]
                        pix = page.get_pixmap(dpi=dpi, alpha=False)
                        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                        del pix

                        sheet_file = masked_dir / f"sheet_{s_idx:04d}.png"
                        if not sheet_file.is_file() or overwrite:
                            cv2.imwrite(str(sheet_file), img_bgr)

                        split_pages = detect_and_split_spread(img_bgr)
                        if len(split_pages) != len(book_pages):
                            if len(split_pages) == 1:
                                split_pages = [(img_bgr, "single")]
                                book_pages = (book_pages[0],)

                        for (sub_img, side), bp in zip(split_pages, book_pages):
                            if self._stop_event.is_set() or thread_errors:
                                break
                            self._check_pause()

                            mask_file = masked_dir / f"page_{bp:04d}_masked.png"
                            if not overwrite and bp in existing_valid_pages:
                                continue

                            deskewed_bgr, _ = deskew_page(sub_img)
                            detections = self.layout_detector.detect(deskewed_bgr)

                            should_skip, reason = section_filter.check_after_detection(
                                bp, total_book_pages, len(detections), gray=cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
                            )
                            if should_skip:
                                pages_done_count += 1
                                existing_valid_pages.add(bp)
                                chk["phases"]["slicing"]["pages_done"] = pages_done_count
                                self._write_checkpoint(pdf_path.stem, chk)

                                cv2.imwrite(str(mask_file), deskewed_bgr)
                                raw_md_file = masked_dir.parent / "3_raw_md" / f"page_{bp:04d}_raw.md"
                                if not raw_md_file.is_file() or overwrite:
                                    raw_md_file.parent.mkdir(parents=True, exist_ok=True)
                                    raw_md_file.write_text(f"## Страница {bp}\n\n<!-- Пропуск: {reason} -->\n", encoding="utf-8")

                                pct_p1 = round((bp / max(1, total_book_pages)) * 100.0, 1)
                                detail_msg = f"Стр. {bp}/{total_book_pages} • Пропуск: {reason}"
                                self._update_phase_progress("phase1", pct_p1, bp, total_book_pages, "running", detail_msg)
                                self._update_hud("Фазы 1-2: Потоковая нарезка и OMR", (bp / total_book_pages) * 25.0, detail_msg)
                                continue

                            masked_img, crops_data = self.layout_detector.mask_page(
                                deskewed_bgr, detections, tag_prefix=f"{pdf_path.stem}_P{bp:04d}"
                            )

                            debug_img = self.layout_detector.render_debug_image(deskewed_bgr, detections)
                            debug_file = masked_dir / f"page_{bp:04d}_debug.png"
                            cv2.imwrite(str(debug_file), debug_img)
                            cv2.imwrite(str(mask_file), masked_img)

                            try:
                                rel_dbg = debug_file.relative_to(self.output_root).as_posix()
                                with self._lock:
                                    self.metrics["current_page_image_url"] = f"/output/{rel_dbg}"
                                    self.metrics["current_page_image_path"] = str(debug_file)
                                if self.on_frame_update is not None:
                                    self.on_frame_update(debug_img, {
                                        "path": str(debug_file),
                                        "stage": "phase1",
                                        "book": pdf_path.stem,
                                        "page": bp,
                                        "title": f"Стр. {bp} ({side}) — BBox разметка",
                                    })
                            except Exception:
                                pass
                            self.log_event("YOLO", f"Стр. {bp} ({side}): найдено {len(crops_data)} станов")

                            new_crops_to_enqueue = []
                            for crop in crops_data:
                                crop_name = f"{crop['stub_id']}.png"
                                crop_path = crops_dir / crop_name
                                if not crop_path.is_file() or overwrite:
                                    cv2.imwrite(str(crop_path), crop["crop_img"])

                                if overwrite or crop["stub_id"] not in existing_valid_abcs:
                                    new_crops_to_enqueue.append(crop_path)

                            with stats_lock:
                                stats["crops_cut"] += len(new_crops_to_enqueue)
                                total_c = max(1, stats["crops_cut"])
                                chk["phases"]["omr"]["total_crops"] = total_c

                            # Push crops into queue immediately without waiting for OMR!
                            for cp in new_crops_to_enqueue:
                                omr_queue.put(cp)

                            pages_done_count += 1
                            existing_valid_pages.add(bp)
                            chk["phases"]["slicing"]["pages_done"] = pages_done_count
                            self._write_checkpoint(pdf_path.stem, chk)

                            pct_p1 = round((bp / max(1, total_book_pages)) * 100.0, 1)
                            detail_msg = f"Стр. {bp}/{total_book_pages} ({side}) • Вырезано: {len(crops_data)}"
                            self._update_phase_progress("phase1", pct_p1, bp, total_book_pages, "running", detail_msg)
                            self._update_hud("Фазы 1-2: Потоковая нарезка и OMR", (bp / total_book_pages) * 25.0, detail_msg)

                    if not self._stop_event.is_set() and not thread_errors:
                        chk["phases"]["slicing"]["completed"] = True
                        chk["phases"]["slicing"]["pages_done"] = total_book_pages
                        self._write_checkpoint(pdf_path.stem, chk)
                        self._update_phase_progress("phase1", 100.0, total_book_pages, total_book_pages, "completed", f"Все {total_book_pages} стр. нарезаны")
                except Exception as ex:
                    thread_errors.append(ex)
                finally:
                    # Immediately unload LayoutDetector (YOLO) from GPU memory to free ~2.0 GB VRAM for OMR
                    if self.layout_detector is not None:
                        try:
                            if hasattr(self.layout_detector, "purge_gpu_memory"):
                                self.layout_detector.purge_gpu_memory()
                        except Exception:
                            pass
                        self.layout_detector = None
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    producer_done.set()
                    omr_queue.put(None)  # Sentinel to wake consumer

            # ---------------- Consumer: OMR Worker ---------------- #
            def _consumer_omr() -> None:
                try:
                    batch_size = 6
                    batch_counter = 0
                    while not self._stop_event.is_set() and not thread_errors:
                        self._check_pause()

                        batch: List[Path] = []
                        try:
                            item = omr_queue.get(timeout=0.15)
                        except queue.Empty:
                            if producer_done.is_set() and omr_queue.empty():
                                break
                            continue

                        if item is None:
                            # Producer reached EOF, sentinel received
                            # Drain any remaining items in safe chunks of batch_size (never unbounded!)
                            drain_items = []
                            while not omr_queue.empty():
                                try:
                                    rem = omr_queue.get_nowait()
                                    if rem is not None:
                                        drain_items.append(rem)
                                except queue.Empty:
                                    break
                            for b_start in range(0, len(drain_items), batch_size):
                                sub_batch = drain_items[b_start : b_start + batch_size]
                                if sub_batch:
                                    saved = self._transcribe_crops_batch_files(crops_dir, sub_batch)
                                    with stats_lock:
                                        stats["crops_done"] += saved
                                        cd = stats["crops_done"]
                                        tc = max(1, stats["crops_cut"])
                                    chk["phases"]["omr"]["crops_done"] = cd
                                    self._write_checkpoint(pdf_path.stem, chk)
                                    pct_p2 = round((cd / tc) * 100.0, 1)
                                    omr_detail = f"Стан {cd}/{tc} (пачка {len(sub_batch)})"
                                    self._update_phase_progress("phase2", pct_p2, cd, tc, "running", omr_detail)
                            break

                        batch.append(item)
                        # Greedily gather up to batch_size without blocking
                        while len(batch) < batch_size:
                            try:
                                nxt = omr_queue.get_nowait()
                                if nxt is None:
                                    producer_done.set()
                                    break
                                batch.append(nxt)
                            except queue.Empty:
                                break

                        if batch:
                            saved = self._transcribe_crops_batch_files(crops_dir, batch)
                            batch_counter += 1
                            if batch_counter % 8 == 0 and torch.cuda.is_available():
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass

                            with stats_lock:
                                stats["crops_done"] += saved
                                cd = stats["crops_done"]
                                tc = max(1, stats["crops_cut"])
                            chk["phases"]["omr"]["crops_done"] = cd
                            self._write_checkpoint(pdf_path.stem, chk)

                            self.log_event("OMR", f"Распознана пачка ({saved} станов, всего: {cd}/{tc})")


                            pct_p2 = round((cd / tc) * 100.0, 1)
                            omr_detail = f"Стан {cd}/{tc} (пачка {len(batch)})"
                            self._update_phase_progress("phase2", pct_p2, cd, tc, "running", omr_detail)

                    if not self._stop_event.is_set() and not thread_errors:
                        final_total_crops = len([f for f in crops_dir.glob("*.png") if not f.name.endswith("_deskew.png")])
                        chk["phases"]["omr"]["total_crops"] = final_total_crops
                        chk["phases"]["omr"]["crops_done"] = final_total_crops
                        chk["phases"]["omr"]["completed"] = True
                        self._write_checkpoint(pdf_path.stem, chk)
                        if final_total_crops > 0:
                            self._update_phase_progress("phase2", 100.0, final_total_crops, final_total_crops, "completed", f"Все {final_total_crops} станов распознаны")
                        else:
                            self._update_phase_progress("phase2", 100.0, 0, 0, "completed", "Нет станов для OMR")
                except Exception as ex:
                    thread_errors.append(ex)

            # Start concurrent threads
            p_thread = threading.Thread(target=_producer_slicing, name="Phase1_Slicing_Producer", daemon=True)
            c_thread = threading.Thread(target=_consumer_omr, name="Phase2_OMR_Consumer", daemon=True)

            p_thread.start()
            c_thread.start()

            # Wait for both workers to finish
            while p_thread.is_alive() or c_thread.is_alive():
                if self._stop_event.is_set() or thread_errors:
                    break
                time.sleep(0.1)

            p_thread.join(timeout=30.0)
            c_thread.join(timeout=30.0)

            if thread_errors:
                raise thread_errors[0]

            if self._stop_event.is_set():
                return False

        return True

    # ---------------- Phase 1 Implementation ---------------- #

    def _phase_1_slicing(
        self, pdf_path: Path, crops_dir: Path, masked_dir: Path, chk: Dict[str, Any], overwrite: bool
    ) -> bool:
        with fitz.open(pdf_path) as doc:
            total_sheets = len(doc)

            # Pre-compute sheet to book page mapping (two-page spreads produce 2 book pages)
            sheet_mapping: List[Tuple[int, ...]] = []
            cur_bp = 1
            for s_idx in range(total_sheets):
                p = doc[s_idx]
                w, h = p.rect.width, p.rect.height
                is_spread = (1.22 <= (w / float(max(1.0, h))) <= 2.2 and h >= 300)
                if is_spread:
                    sheet_mapping.append((cur_bp, cur_bp + 1))
                    cur_bp += 2
                else:
                    sheet_mapping.append((cur_bp,))
                    cur_bp += 1

            total_book_pages = cur_bp - 1
            chk["phases"]["slicing"]["total_pages"] = total_book_pages

            existing_valid_pages = set()
            if not overwrite:
                for f in masked_dir.glob("page_*_masked.png"):
                    m = re.search(r"page_(\d+)_masked", f.stem)
                    if m and f.stat().st_size > 1000:
                        existing_valid_pages.add(int(m.group(1)))

            if not overwrite and len(existing_valid_pages) >= total_book_pages and total_book_pages > 0:
                chk["phases"]["slicing"]["completed"] = True
                chk["phases"]["slicing"]["pages_done"] = total_book_pages
                self._write_checkpoint(pdf_path.stem, chk)
                self._update_phase_progress("phase1", 100.0, total_book_pages, total_book_pages, "completed", f"Все {total_book_pages} стр. готовы")
                self._update_hud("Фаза 1/4: Нарезка уже выполнена на диске", 25.0, f"Все {total_book_pages} стр. готовы")
                return True

            if self.layout_detector is None:
                weights_path = ROOT_DIR / "weights" / "ola-layout-analysis-2.0-2025-03-09.pt"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.layout_detector = LayoutDetector(weights_path=str(weights_path), device=device)

            section_filter = BookSectionFilter(
                skip_front_matter=self.config.get("skip_front_matter", True),
                skip_back_matter=self.config.get("skip_back_matter", True),
            )
            sections = section_filter.analyze_document_sections(pdf_path)

            dpi = int(self.config.get("dpi", 200))
            pages_done_count = len(existing_valid_pages)

            for s_idx in range(total_sheets):
                if self._stop_event.is_set():
                    return False
                self._check_pause()

                book_pages = sheet_mapping[s_idx]
                if not overwrite and all(bp in existing_valid_pages for bp in book_pages):
                    continue

                page = doc[s_idx]
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                del pix

                # Persist original un-split sheet scan for UI inspection / Mode 1 comparison
                sheet_file = masked_dir / f"sheet_{s_idx:04d}.png"
                if not sheet_file.is_file() or overwrite:
                    cv2.imwrite(str(sheet_file), img_bgr)

                # Automated two-page spread splitting along central spine / gutter
                split_pages = detect_and_split_spread(img_bgr)
                if len(split_pages) != len(book_pages):
                    if len(split_pages) == 1:
                        split_pages = [(img_bgr, "single")]
                        book_pages = (book_pages[0],)

                for (sub_img, side), bp in zip(split_pages, book_pages):
                    if self._stop_event.is_set():
                        return False
                    self._check_pause()

                    mask_file = masked_dir / f"page_{bp:04d}_masked.png"
                    if not overwrite and bp in existing_valid_pages:
                        continue

                    # Preprocess: individual book page deskew
                    deskewed_bgr, _ = deskew_page(sub_img)

                    # Layout detection (scale-adaptive, physical staff periodicity verification)
                    detections = self.layout_detector.detect(deskewed_bgr)

                    # Dynamic front/back matter skip
                    should_skip, reason = section_filter.check_after_detection(
                        bp, total_book_pages, len(detections), gray=cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
                    )
                    if should_skip:
                        pages_done_count += 1
                        existing_valid_pages.add(bp)
                        chk["phases"]["slicing"]["pages_done"] = pages_done_count
                        self._write_checkpoint(pdf_path.stem, chk)

                        # Write empty mask placeholder for physical disk tracking
                        cv2.imwrite(str(mask_file), deskewed_bgr)
                        raw_md_file = masked_dir.parent / "3_raw_md" / f"page_{bp:04d}_raw.md"
                        if not raw_md_file.is_file() or overwrite:
                            raw_md_file.parent.mkdir(parents=True, exist_ok=True)
                            raw_md_file.write_text(f"## Страница {bp}\n\n<!-- Пропуск: {reason} -->\n", encoding="utf-8")

                        pct_p1 = round((bp / max(1, total_book_pages)) * 100.0, 1)
                        detail_msg = f"Стр. {bp}/{total_book_pages} • Пропуск: {reason}"
                        self._update_phase_progress("phase1", pct_p1, bp, total_book_pages, "running", detail_msg)
                        self._update_hud(
                            "Фаза 1/4: Нарезка и маскирование",
                            (bp / total_book_pages) * 25.0,
                            detail_msg,
                        )
                        continue

                    # Whiteout and save crops
                    masked_img, crops_data = self.layout_detector.mask_page(
                        deskewed_bgr, detections, tag_prefix=f"{pdf_path.stem}_P{bp:04d}"
                    )

                    # Render and save debug image with YOLO bounding boxes for UI inspection
                    debug_img = self.layout_detector.render_debug_image(deskewed_bgr, detections)
                    debug_file = masked_dir / f"page_{bp:04d}_debug.png"
                    cv2.imwrite(str(debug_file), debug_img)

                    cv2.imwrite(str(mask_file), masked_img)

                    for crop in crops_data:
                        crop_name = f"{crop['stub_id']}.png"
                        crop_path = crops_dir / crop_name
                        if not crop_path.is_file() or overwrite:
                            cv2.imwrite(str(crop_path), crop["crop_img"])

                    pages_done_count += 1
                    existing_valid_pages.add(bp)
                    chk["phases"]["slicing"]["pages_done"] = pages_done_count
                    self._write_checkpoint(pdf_path.stem, chk)

                    pct_p1 = round((bp / max(1, total_book_pages)) * 100.0, 1)
                    detail_msg = f"Стр. {bp}/{total_book_pages} ({side}) • Вырезано станов: {len(crops_data)}"
                    self._update_phase_progress("phase1", pct_p1, bp, total_book_pages, "running", detail_msg)
                    self._update_hud(
                        "Фаза 1/4: Нарезка и маскирование",
                        (bp / total_book_pages) * 25.0,
                        detail_msg,
                    )

        self._update_phase_progress("phase1", 100.0, total_book_pages, total_book_pages, "completed", f"Все {total_book_pages} стр. нарезаны")
        return True

    # ---------------- Phase 2 Implementation ---------------- #

    def _phase_2_omr(self, crops_dir: Path, chk: Dict[str, Any], overwrite: bool) -> bool:
        crop_files = sorted(
            [f for f in crops_dir.glob("*.png") if not f.name.endswith("_deskew.png")],
            key=lambda f: f.name,
        )
        total_crops = len(crop_files)
        chk["phases"]["omr"]["total_crops"] = total_crops

        if total_crops == 0:
            chk["phases"]["omr"]["completed"] = True
            self._update_phase_progress("phase2", 100.0, 0, 0, "completed", "Нет станов для OMR")
            return True

        existing_valid_abcs = set()
        if not overwrite:
            for f in crops_dir.glob("*.abc"):
                if f.stat().st_size > 0:
                    existing_valid_abcs.add(f.stem)

        if not overwrite and len(existing_valid_abcs) >= total_crops:
            chk["phases"]["omr"]["completed"] = True
            chk["phases"]["omr"]["crops_done"] = total_crops
            self._write_checkpoint(crops_dir.parent.name, chk)
            self._update_phase_progress("phase2", 100.0, total_crops, total_crops, "completed", f"Все {total_crops} станов готовы")
            self._update_hud("Фаза 2/4: OMR уже выполнен на диске", 50.0, f"Все {total_crops} станов готовы")
            return True

        if self.omr_engine is None:
            from core.omr_engine import OMREngine

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.omr_engine = OMREngine(
                device=device,
                enable_score_enhancer=self.config.get("enable_score_enhancer", True),
                enable_cugan=self.config.get("enable_cugan_sr", True)
            )

        pending_crops = []
        for crop_file in crop_files:
            if overwrite or crop_file.stem not in existing_valid_abcs:
                pending_crops.append(crop_file)

        crops_done_count = len(existing_valid_abcs)
        batch_size = 6  # Balanced mini-batch size for optimal Tensor Core saturation and low VRAM

        for b_start in range(0, len(pending_crops), batch_size):
            if self._stop_event.is_set():
                return False
            self._check_pause()

            batch_files = pending_crops[b_start:b_start + batch_size]
            saved = self._transcribe_crops_batch_files(crops_dir, batch_files)
            crops_done_count += saved

            # Update checkpoint and HUD once per batch (instead of per single crop)
            chk["phases"]["omr"]["crops_done"] = crops_done_count
            self._write_checkpoint(crops_dir.parent.name, chk)

            curr_idx = min(total_crops, len(existing_valid_abcs) + b_start + len(batch_files))
            pct_p2 = round((curr_idx / max(1, total_crops)) * 100.0, 1)
            detail_msg = f"Стан {curr_idx}/{total_crops} [batch={len(batch_files)}]"
            self._update_phase_progress("phase2", pct_p2, curr_idx, total_crops, "running", detail_msg)
            self._update_hud(
                "Фаза 2/4: Распознавание нот (OMR)",
                25.0 + (curr_idx / max(1, total_crops)) * 25.0,
                detail_msg,
            )

        self._update_phase_progress("phase2", 100.0, total_crops, total_crops, "completed", f"Все {total_crops} станов распознаны")
        chk["phases"]["omr"]["completed"] = True
        self._write_checkpoint(crops_dir.parent.name, chk)
        return True

    # ---------------- Phase 3 Implementation ---------------- #

    def _phase_3_vlm(
        self, masked_dir: Path, raw_md_dir: Path, chk: Dict[str, Any], overwrite: bool
    ) -> bool:
        mask_files = sorted(masked_dir.glob("page_*_masked.png"))
        total_masks = len(mask_files)
        chk["phases"]["vlm"]["total_pages"] = total_masks

        if total_masks == 0:
            chk["phases"]["vlm"]["completed"] = True
            self._update_phase_progress("phase3", 100.0, 0, 0, "completed", "Нет страниц для VLM")
            return True

        existing_valid_mds = set()
        if not overwrite:
            for f in raw_md_dir.glob("page_*_raw.md"):
                if f.stat().st_size > 0:
                    m = re.search(r"page_(\d+)_raw", f.stem)
                    if m:
                        existing_valid_mds.add(m.group(1))

        if not overwrite and len(existing_valid_mds) >= total_masks:
            chk["phases"]["vlm"]["completed"] = True
            chk["phases"]["vlm"]["pages_done"] = total_masks
            self._write_checkpoint(masked_dir.parent.name, chk)
            self._update_phase_progress("phase3", 100.0, total_masks, total_masks, "completed", f"Все {total_masks} стр. готовы")
            self._update_hud("Фаза 3/4: Текст VLM уже извлечен на диске", 90.0, f"Все {total_masks} стр. готовы")
            return True

        # Check if VLM is explicitly skipped by configuration
        if self.config.get("skip_vlm", False):
            self._vlm_fallback_mode(mask_files, raw_md_dir, chk)
            chk["phases"]["vlm"]["completed"] = True
            chk["phases"]["vlm"]["pages_done"] = total_masks
            self._write_checkpoint(masked_dir.parent.name, chk)
            self._update_phase_progress("phase3", 100.0, total_masks, total_masks, "skipped", "Пропущено (skip_vlm=True)")
            self._update_hud("Фаза 3/4: VLM OCR пропущен (skip_vlm=True)", 90.0, "Текстовый VLM отключен в настройках")
            return True

        # Check VLM backend choice
        vlm_backend = self.config.get("vlm_backend", "embedded")
        if vlm_backend == "embedded":
            if not self.embedded_runner.is_server_ready():
                self.log_event("VLM", "Проверка локальных весов модели Qwen...")
                m_info = self.model_manager.resolve_model_files(
                    repo=self.config.get("vlm_model_repo", "lmstudio-community/Qwen3.5-9B-GGUF"),
                    model_file=self.config.get("vlm_model_file", "Qwen3.5-9B-Q4_K_M.gguf"),
                    mmproj_file=self.config.get("vlm_mmproj_file", "mmproj-Qwen3.5-9B-BF16.gguf"),
                )
                if m_info["ready"]:
                    self.log_event("VLM", "Запуск автономного VLM сервера llama-server...")
                    port_cfg = int(self.config.get("vlm_embedded_port", 1234))
                    started = self.embedded_runner.start(
                        model_path=m_info["model_path"],
                        mmproj_path=m_info["mmproj_path"],
                        context_length=int(self.config.get("qwen_context_length", 4096)),
                        parallel=int(self.config.get("qwen_parallel", 1)),
                        flash_attention=bool(self.config.get("qwen_flash_attention", True)),
                        port=port_cfg,
                    )
                    if started:
                        self.lm_client.port = str(self.embedded_runner.port)
                        self.lm_client.base_url = f"http://{self.lm_client.host}:{self.lm_client.port}"
                    else:
                        self.log_event("WARN", "Не удалось запустить автономный llama-server, проверка порта 1234...")
                else:
                    self.log_event("WARN", "Веса VLM не найдены на диске. Требуется загрузка или запуск LM Studio.")

        # Check LM Studio / llama-server availability
        is_online, msg = self.lm_client.check_connection()
        if not is_online:
            with self._lock:
                self.metrics["last_log"] = f"VLM сервер недоступен: {msg}"
            self.log_event("WARN", f"VLM сервер недоступен: {msg}. Запуск fallback...")
            ok = self._vlm_fallback_mode(mask_files, raw_md_dir, chk)
            self._update_phase_progress("phase3", 100.0, total_masks, total_masks, "completed", "Сформировано (fallback)")
            return ok

        # If backend is LM Studio, attempt explicit model loading
        if vlm_backend == "lm_studio":
            try:
                self.lm_client.load_model(
                    model_name=self.config.get("lm_model", "qwen3.5-9b"),
                    context_length=int(self.config.get("qwen_context_length", 4096)),
                    eval_batch_size=int(self.config.get("qwen_eval_batch_size", 2048)),
                    flash_attention=bool(self.config.get("qwen_flash_attention", True)),
                    offload_kv_cache=bool(self.config.get("qwen_offload_kv_cache_to_gpu", True)),
                    parallel=int(self.config.get("qwen_parallel", 1)),
                )
            except Exception as e:
                self.log_event("WARN", f"Предупреждение при загрузке модели: {e}")

        system_prompt = self.config.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        vlm_done_count = len(existing_valid_mds)

        for m_idx, mask_file in enumerate(mask_files, start=1):
            if self._stop_event.is_set():
                return False
            self._check_pause()

            p_num_str = re.search(r"page_(\d+)_masked", mask_file.stem)
            p_num = p_num_str.group(1) if p_num_str else f"{m_idx:04d}"
            raw_md_file = raw_md_dir / f"page_{p_num}_raw.md"

            if not overwrite and p_num in existing_valid_mds:
                continue

            # Update live preview image for current page
            try:
                rel_mask = mask_file.relative_to(self.output_root).as_posix()
                with self._lock:
                    self.metrics["current_page_image_url"] = f"/output/{rel_mask}"
                    self.metrics["current_page_image_path"] = str(mask_file)
                if self.on_frame_update is not None:
                    self.on_frame_update(None, {
                        "path": str(mask_file),
                        "stage": "phase3",
                        "book": pdf_path.stem,
                        "page": page_num,
                        "title": f"Стр. {page_num} — VLM маскированная страница",
                    })
            except Exception:
                pass

            try:
                img_bytes = mask_file.read_bytes()
                detailed = self.lm_client.request_ocr_detailed(
                    image_bytes=img_bytes,
                    system_prompt=system_prompt,
                    model_name=self.config.get("lm_model", "qwen3.5-9b"),
                    temperature=float(self.config.get("lm_temperature", 0.1)),
                    max_tokens=int(self.config.get("lm_max_tokens", 2048)),
                    context_length=int(self.config.get("qwen_context_length", 4096)),
                    max_dim=int(self.config.get("vlm_max_dim", 1600)),
                )
                extracted_text = detailed["text"]
                tok_per_sec = detailed.get("tokens_per_second", 0.0)
                duration = detailed.get("duration", 0.0)
                tok_count = detailed.get("tokens_count", 0)

                with self._lock:
                    self.metrics["vlm_tok_per_sec"] = tok_per_sec
                    self.metrics["last_page_seconds"] = duration
                    self.metrics["last_page_tokens"] = tok_count

                self.log_event("VLM", f"Стр. {p_num}: {tok_count} токенов ({tok_per_sec:.1f} tok/s, {duration:.1f}s)")
                raw_md_file.write_text(self._sanitize_vlm_text(extracted_text), encoding="utf-8")
            except Exception as e:
                self.log_event("ERROR", f"Стр. {p_num}: ошибка VLM ({str(e)})", level="ERROR")
                fallback_txt = f"<!-- LM_STUDIO_ERROR: {str(e)} -->\n\n## Страница {int(p_num)}\n\n[Текст не распознан: ошибка связи с VLM]\n"
                raw_md_file.write_text(fallback_txt, encoding="utf-8")

            vlm_done_count += 1
            chk["phases"]["vlm"]["pages_done"] = vlm_done_count
            self._write_checkpoint(masked_dir.parent.name, chk)

            pct_p3 = round((m_idx / max(1, total_masks)) * 100.0, 1)
            detail_msg = f"Стр. {m_idx}/{total_masks} (Qwen VLM через LM Studio)"
            self._update_phase_progress("phase3", pct_p3, m_idx, total_masks, "running", detail_msg)
            self._update_hud(
                "Фаза 3/4: Текстовый VLM проход",
                55.0 + (m_idx / total_masks) * 35.0,
                detail_msg,
            )

        self._update_phase_progress("phase3", 100.0, total_masks, total_masks, "completed", f"Все {total_masks} стр. обработаны VLM")
        chk["phases"]["vlm"]["completed"] = True
        self._write_checkpoint(masked_dir.parent.name, chk)
        return True

    def _vlm_fallback_mode(self, mask_files: List[Path], raw_md_dir: Path, chk: Dict[str, Any]) -> bool:
        """Creates clean structured markdown files with preserved MUSIC_STUB_ID tags if LM Studio is offline/skipped."""
        crops_dir = raw_md_dir.parent / "1_crops"
        overwrite = self.config.get("overwrite", False)

        for m_idx, mask_file in enumerate(mask_files, start=1):
            p_num_str = re.search(r"page_(\d+)_masked", mask_file.stem)
            p_num = p_num_str.group(1) if p_num_str else f"{m_idx:04d}"
            raw_md_file = raw_md_dir / f"page_{p_num}_raw.md"
            if not raw_md_file.is_file() or overwrite:
                stubs = []
                if crops_dir.is_dir():
                    stubs = sorted([
                        f.stem for f in crops_dir.glob(f"*_P{p_num}_S*.png")
                        if not f.name.endswith("_deskew.png")
                    ])

                lines = [f"## Страница {int(p_num)}\n"]
                if stubs:
                    for stub_id in stubs:
                        lines.append(f"<!-- MUSIC_STUB_ID:{stub_id} -->\n")
                else:
                    lines.append("<!-- Страница без нотного материала -->\n")

                raw_md_file.write_text("\n".join(lines), encoding="utf-8")
        return True

    # ---------------- Phase 4 Implementation ---------------- #

    def _phase_4_assembly(
        self, book_dir: Path, raw_md_dir: Path, crops_dir: Path, final_dir: Path, chk: Dict[str, Any]
    ) -> bool:
        overwrite = self.config.get("overwrite", False)
        complete_book_file = book_dir / f"{book_dir.name}_complete.md"
        if not overwrite and complete_book_file.is_file() and complete_book_file.stat().st_size > 0:
            chk["phases"]["assembly"]["completed"] = True
            self._update_phase_progress("phase4", 100.0, 1, 1, "completed", "Издание уже собрано на диске")
            return True

        raw_files = sorted(raw_md_dir.glob("page_*_raw.md"))
        total_raw = len(raw_files)
        self._update_phase_progress("phase4", 10.0, 0, total_raw, "running", "Сборка Markdown страниц...")

        all_pages_content = []
        injected_stubs = set()

        def inject_music_blocks(match):
            cid = match.group(1).strip()
            injected_stubs.add(cid)
            abc_file = crops_dir / f"{cid}.abc"
            kern_file = crops_dir / f"{cid}.kern"

            blocks = []
            if abc_file.is_file():
                abc = abc_file.read_text(encoding="utf-8", errors="replace").strip()
                if abc and not abc.startswith("% [OMR Conversion Error"):
                    blocks.append(f"```abc\n{abc}\n```")

            if kern_file.is_file():
                raw_kern = kern_file.read_text(encoding="utf-8", errors="replace").strip()
                if raw_kern:
                    try:
                        healed_kern = self.bridge.normalize_humdrum(raw_kern)
                        blocks.append(f"```kern\n{healed_kern}\n```")
                    except Exception:
                        blocks.append(f"```kern\n{raw_kern}\n```")

            if blocks:
                return "\n\n" + "\n\n".join(blocks) + "\n\n"
            return f"\n\n<!-- MUSIC_STUB_ID:{cid} (Ноты не найдены, ожидают OMR) -->\n\n"

        for r_idx, r_file in enumerate(raw_files, start=1):
            p_num_str = re.search(r"page_(\d+)_raw", r_file.stem)
            p_num = p_num_str.group(1) if p_num_str else "0001"
            final_file = final_dir / f"page_{p_num}.md"

            raw_text = r_file.read_text(encoding="utf-8")
            raw_text = self._sanitize_vlm_text(raw_text)

            # Check which stubs were expected for this page from crops_dir
            page_stubs = []
            if crops_dir.is_dir():
                page_stubs = sorted([
                    f.stem for f in crops_dir.glob(f"*_P{p_num}_S*.png")
                    if not f.name.endswith("_deskew.png")
                ])

            injected_stubs.clear()
            final_text = re.sub(r"<!--\s*MUSIC_STUB_ID:\s*(.*?)\s*-->", inject_music_blocks, raw_text)

            # Fallback recovery: if any stubs were omitted by VLM, append them cleanly to the bottom
            missing_stubs = [s for s in page_stubs if s not in injected_stubs]
            if missing_stubs:
                recovered_blocks = []
                for ms in missing_stubs:
                    abc_file = crops_dir / f"{ms}.abc"
                    kern_file = crops_dir / f"{ms}.kern"
                    if abc_file.is_file():
                        abc = abc_file.read_text(encoding="utf-8", errors="replace").strip()
                        if abc and not abc.startswith("% [OMR Conversion Error"):
                            recovered_blocks.append(f"```abc\n{abc}\n```")
                    if kern_file.is_file():
                        raw_kern = kern_file.read_text(encoding="utf-8", errors="replace").strip()
                        if raw_kern:
                            try:
                                healed_kern = self.bridge.normalize_humdrum(raw_kern)
                                recovered_blocks.append(f"```kern\n{healed_kern}\n```")
                            except Exception:
                                recovered_blocks.append(f"```kern\n{raw_kern}\n```")
                if recovered_blocks:
                    final_text += "\n\n<!-- RECOVERED_MUSIC_STUBS -->\n\n" + "\n\n".join(recovered_blocks) + "\n"

            final_file.write_text(final_text, encoding="utf-8")
            try:
                (final_dir / f"page_{int(p_num):04d}_final.md").write_text(final_text, encoding="utf-8")
            except Exception:
                pass
            all_pages_content.append(f"<!-- PAGE {p_num} -->\n" + final_text)

            pct_p4 = round(10.0 + (r_idx / max(1, total_raw)) * 80.0, 1)
            self._update_phase_progress("phase4", pct_p4, r_idx, total_raw, "running", f"Сборка стр. {p_num} ({r_idx}/{total_raw})")

        full_content = "\n\n---\n\n".join(all_pages_content)
        complete_book_file.write_text(full_content, encoding="utf-8")
        chk["phases"]["assembly"]["completed"] = True
        self._update_phase_progress("phase4", 100.0, total_raw, total_raw, "completed", f"Издание {book_dir.name} собрано")
        self.log_event("ASSEMBLY", f"Издание {book_dir.name} успешно собрано ({total_raw} стр.)")
        return True

    @staticmethod
    def _sanitize_vlm_text(text: str) -> str:
        """
        Sanitizes VLM text output:
        1. Fixes HTML entity escapes for stub tags: &lt;!-- MUSIC_STUB_ID:... --&gt; -> <!-- MUSIC_STUB_ID:... -->
        2. Fixes stray backticks around stub tags: `<!-- MUSIC_STUB_ID:... -->` -> <!-- MUSIC_STUB_ID:... -->
        """
        if not text:
            return text
        text = re.sub(r"&lt;!--\s*MUSIC_STUB_ID:([^\s>]+)\s*--&gt;", r"<!-- MUSIC_STUB_ID:\1 -->", text, flags=re.IGNORECASE)
        text = re.sub(r"<!--\s*MUSIC_STUB_ID:([^\s>]+)\s*-->", r"<!-- MUSIC_STUB_ID:\1 -->", text)
        text = re.sub(r"`\s*(<!--\s*MUSIC_STUB_ID:[^\s>]+?\s*-->)\s*`", r"\1", text)
        return text

    # ---------------- Purge VRAM Barrier ---------------- #

    def _purge_vram(self) -> None:
        if self.omr_engine is not None:
            try:
                self.omr_engine.purge_gpu_memory()
            except Exception:
                pass
            self.omr_engine = None

        if self.layout_detector is not None:
            try:
                if hasattr(self.layout_detector, "purge_gpu_memory"):
                    self.layout_detector.purge_gpu_memory()
            except Exception:
                pass
            self.layout_detector = None

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
