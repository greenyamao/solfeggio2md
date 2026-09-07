"""
Fault-Tolerant Multi-File Batch Pipeline Runner.
Sequential Batch Passes: Slicing -> OMR -> VRAM Purge Barrier -> VLM (LM Studio) -> Markdown Assembly.
Resumable via atomic checkpoints (checkpoint.json) across 20+ books.
"""

import gc
import json
import os
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


DEFAULT_CONFIG_FILE = ROOT_DIR / "config.json"
QUEUE_STATE_FILE = ROOT_DIR / "queue_state.json"
DEFAULT_CONFIG = {
    "lm_host": "127.0.0.1",
    "lm_port": "1234",
    "lm_model": "qwen/qwen3.5-9b",
    "lm_temperature": 0.1,
    "lm_max_tokens": 8192,
    "output_dir": str(ROOT_DIR / "output"),
    "dpi": 200,
    "delay": 0.2,
    "smt_max_tokens": 512,
    "qwen_context_length": 16196,
    "qwen_eval_batch_size": 2048,
    "qwen_flash_attention": True,
    "qwen_offload_kv_cache_to_gpu": True,
    "smt_model": "antoniorv6/smt-grandstaff",
    "overwrite": False,
    "smt_device": "cuda" if torch.cuda.is_available() else "cpu",
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
        p_out = Path(cfg_out)
        if not p_out.is_absolute():
            self.output_root = (ROOT_DIR / p_out).resolve()
        else:
            if not p_out.exists() or "Users" in str(p_out):
                self.output_root = (ROOT_DIR / "output").resolve()
            else:
                self.output_root = p_out
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.input_dir = ROOT_DIR / "in"
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
        }

        # Lazy engines
        self.layout_detector: Optional[LayoutDetector] = None
        self.omr_engine = None
        self.lm_client = LMStudioClient(
            host=self.config.get("lm_host", "127.0.0.1"),
            port=self.config.get("lm_port", "1234"),
        )

        # Initial queue population and directory sync
        self._load_existing_queue()

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
            p_out = Path(cfg_out)
            if not p_out.is_absolute():
                self.output_root = (ROOT_DIR / p_out).resolve()
            else:
                if not p_out.exists() or "Users" in str(p_out):
                    self.output_root = (ROOT_DIR / "output").resolve()
                else:
                    self.output_root = p_out
            self.output_root.mkdir(parents=True, exist_ok=True)
            self.input_dir = ROOT_DIR / "in"
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

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            m = dict(self.metrics)
            m["total_books"] = len(self.queue)
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
                self.metrics["current_phase_name"] = "Очередь полностью обработана"
                self.metrics["current_item_detail"] = "Все задачи завершены"

        except Exception as e:
            with self._lock:
                self.metrics["current_phase_name"] = "Ошибка в конвейере"
                self.metrics["last_log"] = f"Исключение: {str(e)}"
        finally:
            self._purge_vram()
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

        # ---------------- Phase 1: Slicing & Masking ---------------- #
        if not chk["phases"]["slicing"]["completed"] or overwrite:
            self._update_hud("Фаза 1/4: Нарезка и маскирование верстки (YOLO OLA v2.0)", 0.0)
            ok = self._phase_1_slicing(pdf_path, crops_dir, masked_dir, chk, overwrite)
            if not ok:
                return False
            chk["phases"]["slicing"]["completed"] = True
            self._write_checkpoint(book_title, chk)

        # ---------------- Inter-phase Memory Purge Barrier ---------------- #
        self._purge_vram()

        # ---------------- Phase 2: OMR Music Recognition ---------------- #
        if not chk["phases"]["omr"]["completed"] or overwrite:
            self._update_hud("Фаза 2/4: Оптическое распознавание нот (Transcoda-59M)", 25.0)
            ok = self._phase_2_omr(crops_dir, chk, overwrite)
            if not ok:
                return False
            chk["phases"]["omr"]["completed"] = True
            self._write_checkpoint(book_title, chk)

        # ---------------- VRAM Purge Barrier ---------------- #
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

    def _update_hud(self, phase_name: str, book_pct: float, detail: str = "") -> None:
        with self._lock:
            self.metrics["current_phase_name"] = phase_name
            self.metrics["book_progress_pct"] = round(book_pct, 1)
            if detail:
                self.metrics["current_item_detail"] = detail

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

                        pct = (bp / total_book_pages) * 25.0
                        self._update_hud(
                            "Фаза 1/4: Нарезка и маскирование",
                            pct,
                            f"Стр. {bp}/{total_book_pages} • Пропуск: {reason}",
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

                        # Normalize and dewarp staff for OMR (skip if already on disk)
                        deskew_name = f"{crop['stub_id']}_deskew.png"
                        deskew_path = crops_dir / deskew_name
                        if not deskew_path.is_file() or overwrite:
                            dewarped_bgr, _, _ = normalize_staff_crop(
                                crop["crop_img"], notation_class=crop.get("class", "staff")
                            )
                            cv2.imwrite(str(deskew_path), dewarped_bgr)

                    pages_done_count += 1
                    existing_valid_pages.add(bp)
                    chk["phases"]["slicing"]["pages_done"] = pages_done_count
                    self._write_checkpoint(pdf_path.stem, chk)

                    pct = (bp / total_book_pages) * 25.0
                    self._update_hud(
                        "Фаза 1/4: Нарезка и маскирование",
                        pct,
                        f"Стр. {bp}/{total_book_pages} ({side}) • Вырезано станов: {len(crops_data)}",
                    )

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
            self._update_hud("Фаза 2/4: OMR уже выполнен на диске", 50.0, f"Все {total_crops} станов готовы")
            return True

        if self.omr_engine is None:
            from core.omr_engine import OMREngine

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.omr_engine = OMREngine(device=device)

        pending_crops = []
        for crop_file in crop_files:
            if overwrite or crop_file.stem not in existing_valid_abcs:
                pending_crops.append(crop_file)

        crops_done_count = len(existing_valid_abcs)
        batch_size = 4  # Balanced mini-batch size for optimal Tensor Core saturation and low VRAM

        for b_start in range(0, len(pending_crops), batch_size):
            if self._stop_event.is_set():
                return False
            self._check_pause()

            batch_files = pending_crops[b_start:b_start + batch_size]
            batch_crops_bgr = []
            batch_classes = []
            batch_titles = []
            valid_batch_files = []

            for cf in batch_files:
                deskew_file = crops_dir / f"{cf.stem}_deskew.png"
                crop_path_to_read = deskew_file if deskew_file.is_file() else cf
                crop_bgr = cv2.imread(str(crop_path_to_read))
                if crop_bgr is None:
                    continue

                cls_name = "staff"
                if "grand_staff" in cf.stem:
                    cls_name = "grand_staff"
                elif "system" in cf.stem:
                    cls_name = "system"

                batch_crops_bgr.append(crop_bgr)
                batch_classes.append(cls_name)
                batch_titles.append(cf.stem)
                valid_batch_files.append(cf)

            if not batch_crops_bgr:
                continue

            results = self.omr_engine.transcribe_crops_batch(
                crops_bgr=batch_crops_bgr,
                notation_classes=batch_classes,
                titles=batch_titles
            )

            for cf, res in zip(valid_batch_files, results):
                abc_content = res.get("abc", "")
                if abc_content:
                    abc_file = crops_dir / f"{cf.stem}.abc"
                    abc_file.write_text(abc_content, encoding="utf-8")
                crops_done_count += 1

            # Update checkpoint and HUD once per batch (instead of per single crop)
            chk["phases"]["omr"]["crops_done"] = crops_done_count
            self._write_checkpoint(crops_dir.parent.name, chk)

            curr_idx = min(total_crops, len(existing_valid_abcs) + b_start + len(valid_batch_files))
            pct = 25.0 + (curr_idx / max(1, total_crops)) * 25.0
            last_cls = batch_classes[-1] if batch_classes else "staff"
            last_model = results[-1].get("model_used", "OMR") if results else "OMR"
            self._update_hud(
                "Фаза 2/4: Распознавание нот (OMR)",
                pct,
                f"Стан {curr_idx}/{total_crops} ({last_cls}) • {last_model} [batch={len(valid_batch_files)}]",
            )

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
            self._update_hud("Фаза 3/4: Текст VLM уже извлечен на диске", 90.0, f"Все {total_masks} стр. готовы")
            return True

        # Check if VLM is explicitly skipped by configuration
        if self.config.get("skip_vlm", False):
            self._vlm_fallback_mode(mask_files, raw_md_dir, chk)
            chk["phases"]["vlm"]["completed"] = True
            chk["phases"]["vlm"]["pages_done"] = total_masks
            self._write_checkpoint(masked_dir.parent.name, chk)
            self._update_hud("Фаза 3/4: VLM OCR пропущен (skip_vlm=True)", 90.0, "Текстовый VLM отключен в настройках")
            return True

        # Check LM Studio availability
        is_online, msg = self.lm_client.check_connection()
        if not is_online:
            with self._lock:
                self.metrics["last_log"] = f"LM Studio недоступен: {msg}"
            return self._vlm_fallback_mode(mask_files, raw_md_dir, chk)

        # Attempt to load model
        try:
            self.lm_client.load_model(
                model_name=self.config.get("lm_model", "qwen/qwen3.5-9b"),
                context_length=int(self.config.get("qwen_context_length", 16196)),
                eval_batch_size=int(self.config.get("qwen_eval_batch_size", 2048)),
                flash_attention=bool(self.config.get("qwen_flash_attention", True)),
                offload_kv_cache=bool(self.config.get("qwen_offload_kv_cache_to_gpu", True)),
            )
        except Exception as e:
            with self._lock:
                self.metrics["last_log"] = f"Предупреждение при загрузке модели: {e}"

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

            try:
                img_bytes = mask_file.read_bytes()
                extracted_text = self.lm_client.request_ocr(
                    image_bytes=img_bytes,
                    system_prompt=system_prompt,
                    model_name=self.config.get("lm_model", "default"),
                    temperature=float(self.config.get("lm_temperature", 0.1)),
                    max_tokens=int(self.config.get("lm_max_tokens", 8192)),
                    context_length=int(self.config.get("qwen_context_length", 16196)),
                )
                raw_md_file.write_text(self._sanitize_vlm_text(extracted_text), encoding="utf-8")
            except Exception as e:
                fallback_txt = f"<!-- LM_STUDIO_ERROR: {str(e)} -->\n\n## Страница {int(p_num)}\n\n[Текст не распознан: ошибка связи с VLM]\n"
                raw_md_file.write_text(fallback_txt, encoding="utf-8")

            vlm_done_count += 1
            chk["phases"]["vlm"]["pages_done"] = vlm_done_count
            self._write_checkpoint(masked_dir.parent.name, chk)

            pct = 55.0 + (m_idx / total_masks) * 35.0
            self._update_hud(
                "Фаза 3/4: Текстовый VLM проход",
                pct,
                f"Стр. {m_idx}/{total_masks} (Qwen VLM через LM Studio)",
            )

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
            return True

        raw_files = sorted(raw_md_dir.glob("page_*_raw.md"))
        all_pages_content = []

        injected_stubs = set()

        def inject_abc(match):
            cid = match.group(1).strip()
            injected_stubs.add(cid)
            abc_file = crops_dir / f"{cid}.abc"
            if abc_file.is_file():
                abc = abc_file.read_text(encoding="utf-8").strip()
                return f"\n\n```abc\n{abc}\n```\n\n"
            return f"\n\n% [Ноты {cid} не найдены]\n\n"

        for r_file in raw_files:
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
            final_text = re.sub(r"<!--\s*MUSIC_STUB_ID:\s*(.*?)\s*-->", inject_abc, raw_text)

            # Fallback recovery: if any stubs were omitted by VLM, append them cleanly to the bottom
            missing_stubs = [s for s in page_stubs if s not in injected_stubs]
            if missing_stubs:
                recovered_blocks = []
                for ms in missing_stubs:
                    abc_file = crops_dir / f"{ms}.abc"
                    if abc_file.is_file():
                        abc = abc_file.read_text(encoding="utf-8").strip()
                        recovered_blocks.append(f"\n\n```abc\n{abc}\n```\n")
                if recovered_blocks:
                    final_text += "\n\n<!-- RECOVERED_MUSIC_STUBS -->\n" + "\n".join(recovered_blocks)

            final_file.write_text(final_text, encoding="utf-8")
            all_pages_content.append(f"<!-- PAGE {p_num} -->\n" + final_text)

        full_content = "\n\n---\n\n".join(all_pages_content)
        complete_book_file.write_text(full_content, encoding="utf-8")
        chk["phases"]["assembly"]["completed"] = True
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
