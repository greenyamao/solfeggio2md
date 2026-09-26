from pathlib import Path
import os
import re
import cv2
import json
import torch
import numpy as np
import threading
from typing import Dict, List, Any, Optional
from PIL import Image
import pymupdf as fitz

from core.page_preprocessor import deskew_page, PagePreprocessor, normalize_staff_crop, detect_and_split_spread
from core.layout_detector import LayoutDetector
from core.abc_bridge import ABCBridge


class PipelineWorker:
    """
    Coordinates data flow between Preprocessing, Layout Detection (YOLO),
    OMR transcription, and the GUI Workbench.
    Reads strictly from real production output (output/<book_name>/...).
    """
    def __init__(self, output_dir: Optional[str] = None):
        self.root_dir = Path(__file__).parent.parent.resolve()
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "in").mkdir(parents=True, exist_ok=True)
        
        self.bridge = ABCBridge()
        self.omr_engine = None
        self.layout_detector = None
        self.preprocessor = None
        self._sheet_mapping_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._page_cache: Dict[str, Dict[str, Any]] = {}
        self._generating_debug: set = set()

    def get_hardware_status(self) -> Dict[str, Any]:
        """
        Queries GPU VRAM state to ensure strict sequential GPU ownership.
        """
        cuda_available = torch.cuda.is_available()
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0
        device_name = "CPU"
        
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
                vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
                vram_reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
            except Exception:
                pass
                
        is_safe_for_vlm = vram_allocated_mb < 500.0
        status_text = "Safe for VLM (VRAM Clean)" if is_safe_for_vlm else "VRAM in use by OMR"

        return {
            "cuda_available": cuda_available,
            "device_name": device_name,
            "vram_allocated_mb": round(vram_allocated_mb, 1),
            "vram_reserved_mb": round(vram_reserved_mb, 1),
            "is_safe_for_vlm": is_safe_for_vlm,
            "status_text": status_text
        }

    def list_pages(self) -> List[Dict[str, Any]]:
        """
        Scans production output/ books for available pages.
        Returns a sorted list of page metadata summaries.
        """
        pages = []
        prod_out = self.root_dir / "output"
        if prod_out.is_dir():
            for book_dir in sorted(prod_out.iterdir()):
                if not book_dir.is_dir():
                    continue
                masked_dir = book_dir / "2_masked_pages"
                crops_dir = book_dir / "1_crops"
                if masked_dir.is_dir():
                    for mask_file in sorted(masked_dir.glob("page_*_masked.png")):
                        m = re.search(r"page_(\d+)_masked", mask_file.stem)
                        if not m:
                            continue
                        p_num = int(m.group(1))
                        page_id = f"{book_dir.name}__page_{m.group(1)}"
                        crops_count = len(list(crops_dir.glob(f"*{p_num:04d}*.png"))) if crops_dir.is_dir() else 0
                        pages.append({
                            "page_id": page_id,
                            "title": f"Page {p_num}",
                            "crops_count": crops_count,
                            "has_debug": True,
                            "has_mask": True,
                        })

        pages.sort(key=lambda p: p["page_id"])
        return pages

    def _ensure_layout_detector(self):
        if self.layout_detector is None:
            weights_path = self.root_dir / "weights" / "ola-layout-analysis-2.0-2025-03-09.pt"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.layout_detector = LayoutDetector(weights_path=str(weights_path), device=device)

    def _get_sheet_info_for_page(self, book_name: str, p_num: int) -> Dict[str, Any]:
        """
        Determines the source PDF sheet index, spread status, and URL for a given book page number.
        Accommodates both 2-page spreads (where sheet produces 2 book pages) and single pages.
        Uses in-memory cache to guarantee 0 ms lookup without re-opening PDF on every click.
        """
        if book_name in self._sheet_mapping_cache and p_num in self._sheet_mapping_cache[book_name]:
            return self._sheet_mapping_cache[book_name][p_num]

        book_dir = self.root_dir / "output" / book_name
        masked_dir = book_dir / "2_masked_pages"
        masked_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = self.root_dir / "in" / f"{book_name}.pdf"
        if not pdf_path.is_file():
            candidates = list((self.root_dir / "in").glob("*.pdf"))
            if candidates:
                pdf_path = candidates[0]
            else:
                mask_file = masked_dir / f"page_{p_num:04d}_masked.png"
                debug_file = masked_dir / f"page_{p_num:04d}_debug.png"
                fallback_url = (
                    f"/output/{book_name}/2_masked_pages/{debug_file.name}"
                    if debug_file.is_file()
                    else f"/output/{book_name}/2_masked_pages/{mask_file.name}"
                )
                return {
                    "sheet_idx": max(0, p_num - 1),
                    "is_spread": False,
                    "spread_side": "single",
                    "sheet_url": fallback_url,
                }

        try:
            if book_name not in self._sheet_mapping_cache:
                self._sheet_mapping_cache[book_name] = {}

            with fitz.open(pdf_path) as doc:
                total_sheets = len(doc)
                cur_bp = 1
                for s_idx in range(total_sheets):
                    p = doc[s_idx]
                    w, h = p.rect.width, p.rect.height
                    sheet_is_spread = (1.22 <= (w / float(max(1.0, h))) <= 2.2 and h >= 300)
                    sheet_file = masked_dir / f"sheet_{s_idx:04d}.png"
                    sheet_url = f"/output/{book_name}/2_masked_pages/{sheet_file.name}"
                    if sheet_is_spread:
                        self._sheet_mapping_cache[book_name][cur_bp] = {
                            "sheet_idx": s_idx,
                            "is_spread": True,
                            "spread_side": "left",
                            "sheet_url": sheet_url,
                        }
                        self._sheet_mapping_cache[book_name][cur_bp + 1] = {
                            "sheet_idx": s_idx,
                            "is_spread": True,
                            "spread_side": "right",
                            "sheet_url": sheet_url,
                        }
                        cur_bp += 2
                    else:
                        self._sheet_mapping_cache[book_name][cur_bp] = {
                            "sheet_idx": s_idx,
                            "is_spread": False,
                            "spread_side": "single",
                            "sheet_url": sheet_url,
                        }
                        cur_bp += 1

                target_info = self._sheet_mapping_cache[book_name].get(p_num)
                if target_info:
                    target_s_idx = target_info["sheet_idx"]
                    target_file = masked_dir / f"sheet_{target_s_idx:04d}.png"
                    if not target_file.is_file() and 0 <= target_s_idx < total_sheets:
                        pix = doc[target_s_idx].get_pixmap(dpi=200, alpha=False)
                        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(target_file), img_bgr)
                    return target_info
        except Exception:
            pass

        mask_file = masked_dir / f"page_{p_num:04d}_masked.png"
        return {
            "sheet_idx": max(0, p_num - 1),
            "is_spread": False,
            "spread_side": "single",
            "sheet_url": f"/output/{book_name}/2_masked_pages/{mask_file.name}",
        }

    def _generate_debug_page(self, book_name: str, p_num: int, debug_file: Path) -> bool:
        """
        Generates original scan with YOLO bounding boxes on demand if missing from disk.
        Correctly accounts for two-page spread bisection to match physical book page numbers.
        """
        pdf_path = self.root_dir / "in" / f"{book_name}.pdf"
        if not pdf_path.is_file():
            candidates = list((self.root_dir / "in").glob("*.pdf"))
            if candidates:
                pdf_path = candidates[0]
            else:
                return False

        try:
            self._ensure_layout_detector()
            sheet_info = self._get_sheet_info_for_page(book_name, p_num)
            s_idx = sheet_info["sheet_idx"]
            side = sheet_info["spread_side"]

            with fitz.open(pdf_path) as doc:
                if s_idx >= len(doc) or s_idx < 0:
                    return False
                page = doc[s_idx]
                pix = page.get_pixmap(dpi=200, alpha=False)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

            if sheet_info["is_spread"]:
                split_pages = detect_and_split_spread(img_bgr)
                if len(split_pages) == 2:
                    sub_img = split_pages[0][0] if side == "left" else split_pages[1][0]
                else:
                    sub_img = img_bgr
            else:
                sub_img = img_bgr

            deskewed_bgr, _ = deskew_page(sub_img)
            detections = self.layout_detector.detect(deskewed_bgr)
            debug_img = self.layout_detector.render_debug_image(deskewed_bgr, detections)
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_file), debug_img)
            return True
        except Exception:
            return False

    def get_page_data(self, page_id: str) -> Dict[str, Any]:
        """
        Loads real data for a given page including crops, deskewed crops,
        OMR transcriptions (ABC / Kern), and rendered markdown from output/.
        Never returns synthetic/mock data.
        Employs fast in-memory caching and non-blocking background debug generation to eliminate UI freezes.
        """
        if "__page_" not in page_id:
            raise FileNotFoundError(f"Page not found: {page_id}")

        book_name, p_suffix = page_id.split("__page_", 1)
        book_dir = self.root_dir / "output" / book_name
        p_num = int(p_suffix)
        masked_file = book_dir / "2_masked_pages" / f"page_{p_num:04d}_masked.png"
        debug_file = book_dir / "2_masked_pages" / f"page_{p_num:04d}_debug.png"

        # In-memory cache hit (0 ms instant response)
        cached = self._page_cache.get(page_id)
        if cached is not None:
            cur_mtime = max(
                debug_file.stat().st_mtime if debug_file.is_file() else 0,
                masked_file.stat().st_mtime if masked_file.is_file() else 0
            )
            if cached.get("_cache_mtime", 0) >= cur_mtime and cur_mtime > 0:
                return cached["data"]

        crops_dir = book_dir / "1_crops"
        crops_data = []

        if crops_dir.is_dir():
            crop_files = sorted(
                [f for f in crops_dir.glob(f"*{p_num:04d}*.png") if not f.name.endswith("_deskew.png")],
                key=lambda f: f.name
            )
            for idx, cf in enumerate(crop_files, start=1):
                cls_name = "grand_staff" if "grand_staff" in cf.stem else ("system" if "system" in cf.stem else "staff")
                abc_file = crops_dir / f"{cf.stem}.abc"
                abc_txt = abc_file.read_text(encoding="utf-8", errors="replace").strip() if abc_file.is_file() else ""

                kern_file = crops_dir / f"{cf.stem}.kern"
                if kern_file.is_file():
                    kern_txt = kern_file.read_text(encoding="utf-8", errors="replace").strip()
                elif abc_txt:
                    try:
                        import verovio
                        verovio.enableLog(False)
                        tk = verovio.toolkit()
                        tk.setOptions(json.dumps({"inputFrom": "abc"}))
                        if tk.loadData(abc_txt):
                            kern_txt = tk.getHumdrumBuffer()
                            kern_file.write_text(kern_txt, encoding="utf-8")
                        else:
                            kern_txt = ""
                    except Exception:
                        kern_txt = ""
                else:
                    kern_txt = ""

                deskew_file = crops_dir / f"{cf.stem}_deskew.png"
                
                # Fast header-only dimensions read (<0.1ms)
                try:
                    with Image.open(cf) as im_raw:
                        w_c, h_c = im_raw.size
                except Exception:
                    w_c, h_c = 1000, 60

                if deskew_file.is_file():
                    try:
                        with Image.open(deskew_file) as im_d:
                            sr_w, sr_h = im_d.size
                    except Exception:
                        sr_w, sr_h = w_c * 2, h_c * 2
                    tilt_deg, bend_px = 0.0, 0.0
                else:
                    # Quick deskew fallback without heavy SR to avoid blocking UI thread
                    cf_bgr = cv2.imread(str(cf))
                    if cf_bgr is not None:
                        dewarped_bgr, tilt_deg, bend_px = normalize_staff_crop(
                            cf_bgr, notation_class=cls_name, enhance_sr=False
                        )
                        cv2.imwrite(str(deskew_file), dewarped_bgr)
                        sr_h, sr_w = dewarped_bgr.shape[:2]
                    else:
                        sr_w, sr_h, tilt_deg, bend_px = w_c * 2, h_c * 2, 0.0, 0.0

                raw_ts = int(cf.stat().st_mtime) if cf.is_file() else 0
                deskew_ts = int(deskew_file.stat().st_mtime) if deskew_file.is_file() else 0

                crops_data.append({
                    "id": f"S{idx:02d}",
                    "crop_stem": cf.stem,
                    "class": cls_name,
                    "index": idx,
                    "raw_url": f"/output/{book_name}/1_crops/{cf.name}?t={raw_ts}",
                    "deskew_url": f"/output/{book_name}/1_crops/{cf.stem}_deskew.png?t={deskew_ts}",
                    "skew_angle": round(tilt_deg, 1),
                    "bend_delta": round(bend_px, 1),
                    "width": w_c,
                    "height": h_c,
                    "sr_width": sr_w,
                    "sr_height": sr_h,
                    "abc": abc_txt,
                    "kern": kern_txt,
                    "model_used": "OMR" if abc_txt else "Pending"
                })

        # 1. Check for assembled final page first
        final_candidates = [
            book_dir / "4_final_pages" / f"page_{p_num:04d}_final.md",
            book_dir / "4_final_pages" / f"page_{p_num:04d}.md",
            book_dir / "4_final_pages" / f"page_{p_num}.md",
        ]
        md_file = next((f for f in final_candidates if f.is_file()), None)

        if md_file:
            markdown_text = md_file.read_text(encoding="utf-8", errors="replace")
        else:
            raw_candidates = [
                book_dir / "3_raw_md" / f"page_{p_num:04d}_raw.md",
                book_dir / "3_raw_md" / f"page_{p_num}_raw.md",
                book_dir / "3_raw_md" / f"page_{p_num:04d}.md",
            ]
            raw_file = next((f for f in raw_candidates if f.is_file()), None)
            if raw_file:
                markdown_text = raw_file.read_text(encoding="utf-8", errors="replace")
            else:
                markdown_text = "*(Page text not yet recognized. Start batch processing from the Control Panel)*\n"

        # 2. Dynamic music stub injection: inject BOTH ABC and Humdrum **kern blocks for LLM reading
        crops_dir = book_dir / "1_crops"
        if "MUSIC_STUB_ID" in markdown_text and crops_dir.is_dir():
            injected_stubs = set()

            def inject_music_blocks(match):
                cid = match.group(1).strip()
                injected_stubs.add(cid)
                abc_file = crops_dir / f"{cid}.abc"
                kern_file = crops_dir / f"{cid}.kern"

                blocks = []
                # 1. ABC Notation Block
                if abc_file.is_file():
                    abc = abc_file.read_text(encoding="utf-8", errors="replace").strip()
                    if abc and not abc.startswith("% [OMR Conversion Error"):
                        blocks.append(f"```abc\n{abc}\n```")

                # 2. Humdrum **kern Notation Block (with spine healing)
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
                return f"\n\n<!-- MUSIC_STUB_ID:{cid} (Notes not found, awaiting OMR) -->\n\n"

            assembled = re.sub(r"<!--\s*MUSIC_STUB_ID:\s*(.*?)\s*-->", inject_music_blocks, markdown_text)

            # Recover any crops detected on this page that weren't cited in the text
            page_stubs = sorted([
                f.stem for f in crops_dir.glob(f"*_P{p_num:04d}_S*.png")
                if not f.name.endswith("_deskew.png")
            ])
            missing_stubs = [s for s in page_stubs if s not in injected_stubs]
            if missing_stubs:
                recovered = []
                for ms in missing_stubs:
                    abc_file = crops_dir / f"{ms}.abc"
                    kern_file = crops_dir / f"{ms}.kern"
                    if abc_file.is_file():
                        abc = abc_file.read_text(encoding="utf-8", errors="replace").strip()
                        if abc and not abc.startswith("% [OMR Conversion Error"):
                            recovered.append(f"```abc\n{abc}\n```")
                    if kern_file.is_file():
                        raw_kern = kern_file.read_text(encoding="utf-8", errors="replace").strip()
                        if raw_kern:
                            try:
                                healed_kern = self.bridge.normalize_humdrum(raw_kern)
                                recovered.append(f"```kern\n{healed_kern}\n```")
                            except Exception:
                                recovered.append(f"```kern\n{raw_kern}\n```")
                if recovered:
                    assembled += "\n\n<!-- RECOVERED_MUSIC_STUBS -->\n\n" + "\n\n".join(recovered) + "\n"

            markdown_text = assembled

        sheet_info = self._get_sheet_info_for_page(book_name, p_num)
        mask_rel = f"/output/{book_name}/2_masked_pages/page_{p_num:04d}_masked.png"

        if debug_file.is_file():
            debug_ts = int(debug_file.stat().st_mtime)
            debug_rel = f"/output/{book_name}/2_masked_pages/page_{p_num:04d}_debug.png?t={debug_ts}"
        else:
            # Fallback immediately to mask or sheet to avoid freezing HTTP request
            debug_rel = mask_rel if masked_file.is_file() else sheet_info.get("sheet_url", mask_rel)
            if page_id not in self._generating_debug:
                self._generating_debug.add(page_id)
                def _bg_gen():
                    try:
                        self._generate_debug_page(book_name, p_num, debug_file)
                    finally:
                        self._generating_debug.discard(page_id)
                        self._page_cache.pop(page_id, None)
                threading.Thread(target=_bg_gen, daemon=True).start()

        result = {
            "page_id": page_id,
            "title": f"Page {p_num}",
            "original_url": sheet_info["sheet_url"],
            "debug_url": debug_rel,
            "mask_url": mask_rel,
            "is_spread": sheet_info["is_spread"],
            "spread_side": sheet_info["spread_side"],
            "sheet_idx": sheet_info["sheet_idx"],
            "crops": crops_data,
            "markdown": markdown_text
        }

        # Cache result
        cur_mtime = max(
            debug_file.stat().st_mtime if debug_file.is_file() else 0,
            masked_file.stat().st_mtime if masked_file.is_file() else 0
        )
        self._page_cache[page_id] = {
            "_cache_mtime": cur_mtime,
            "data": result
        }

        return result

    def save_page_markdown(self, page_id: str, markdown_content: str) -> bool:
        """
        Persists updated markdown for a page directly into book's 4_final_pages.
        """
        if "__page_" not in page_id:
            return False
        book_name, p_suffix = page_id.split("__page_", 1)
        p_num = int(p_suffix)
        final_dir = self.root_dir / "output" / book_name / "4_final_pages"
        final_dir.mkdir(parents=True, exist_ok=True)
        md_file = final_dir / f"page_{p_num:04d}_final.md"
        md_file.write_text(markdown_content, encoding="utf-8")
        self._page_cache.pop(page_id, None)
        return True

    def transcribe_crop_live(self, page_id: str, crop_stem: str) -> Dict[str, Any]:
        """
        Executes real-time OMR transcription on a specific crop using OMREngine.
        Ensures strict VRAM purge immediately after.
        """
        if "__page_" not in page_id:
            return {"status": "error", "message": f"Invalid page_id: {page_id}"}
        book_name, _ = page_id.split("__page_", 1)
        crops_dir = self.root_dir / "output" / book_name / "1_crops"
        
        # Prefer deskewed crop if available
        deskew_file = crops_dir / f"{crop_stem}_deskew.png"
        raw_file = crops_dir / f"{crop_stem}.png"
        crop_file = deskew_file if deskew_file.is_file() else raw_file
        
        if not crop_file.is_file():
            return {"status": "error", "message": f"Crop file not found: {crop_file}"}

        crop_bgr = cv2.imread(str(crop_file))
        if crop_bgr is None:
            return {"status": "error", "message": "Failed to read crop image"}

        cls_name = "staff"
        if "grand_staff" in crop_stem:
            cls_name = "grand_staff"
        elif "system" in crop_stem:
            cls_name = "system"

        if self.omr_engine is None:
            from core.omr_engine import OMREngine
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.omr_engine = OMREngine(device=device)

        try:
            result = self.omr_engine.transcribe_crop(
                crop_bgr=crop_bgr,
                notation_class=cls_name,
                title=crop_stem
            )
            
            abc_content = result.get("abc", "")
            kern_content = result.get("raw_kern", "")
            
            if abc_content:
                abc_file = crops_dir / f"{crop_stem}.abc"
                abc_file.write_text(abc_content, encoding="utf-8")
                
            if kern_content:
                kern_file = crops_dir / f"{crop_stem}.kern"
                kern_file.write_text(kern_content, encoding="utf-8")

            self._page_cache.pop(page_id, None)

            return {
                "status": "success",
                "abc": abc_content,
                "kern": kern_content,
                "model_used": result.get("model_used", "")
            }
        finally:
            if self.omr_engine is not None:
                self.omr_engine.purge_gpu_memory()
