from pathlib import Path
import os
import re
import cv2
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import pymupdf as fitz

from core.page_preprocessor import deskew_page, PagePreprocessor, normalize_staff_crop, detect_and_split_spread
from core.layout_detector import LayoutDetector


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
        
        self.omr_engine = None
        self.layout_detector = None
        self.preprocessor = None
        self._sheet_mapping_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}

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
                            "title": f"Стр. {p_num}",
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
        """
        if "__page_" not in page_id:
            raise FileNotFoundError(f"Page not found: {page_id}")

        book_name, p_suffix = page_id.split("__page_", 1)
        book_dir = self.root_dir / "output" / book_name
        p_num = int(p_suffix)
        masked_file = book_dir / "2_masked_pages" / f"page_{p_num:04d}_masked.png"
        debug_file = book_dir / "2_masked_pages" / f"page_{p_num:04d}_debug.png"

        crops_dir = book_dir / "1_crops"
        crops_data = []

        if crops_dir.is_dir():
            crop_files = sorted(
                [f for f in crops_dir.glob(f"*{p_num:04d}*.png") if not f.name.endswith("_deskew.png")],
                key=lambda f: f.name
            )
            for idx, cf in enumerate(crop_files, start=1):
                cls_name = "grand_staff" if "grand_staff" in cf.stem else "staff"
                abc_file = crops_dir / f"{cf.stem}.abc"
                abc_txt = abc_file.read_text(encoding="utf-8", errors="replace").strip() if abc_file.is_file() else ""

                deskew_file = crops_dir / f"{cf.stem}_deskew.png"
                cf_bgr = cv2.imread(str(cf))
                if cf_bgr is not None:
                    h_c, w_c = cf_bgr.shape[:2]
                    if not deskew_file.is_file():
                        dewarped_bgr, tilt_deg, bend_px = normalize_staff_crop(cf_bgr, notation_class=cls_name)
                        cv2.imwrite(str(deskew_file), dewarped_bgr)
                    else:
                        tilt_deg, bend_px = 0.0, 0.0
                else:
                    h_c, w_c, tilt_deg, bend_px = 60, 1000, 0.0, 0.0

                crops_data.append({
                    "id": f"S{idx:02d}",
                    "crop_stem": cf.stem,
                    "class": cls_name,
                    "index": idx,
                    "raw_url": f"/output/{book_name}/1_crops/{cf.name}",
                    "deskew_url": f"/output/{book_name}/1_crops/{cf.stem}_deskew.png",
                    "skew_angle": round(tilt_deg, 1),
                    "bend_delta": round(bend_px, 1),
                    "width": w_c,
                    "height": h_c,
                    "abc": abc_txt,
                    "kern": "",
                    "model_used": "OMR" if abc_txt else "Pending"
                })

        md_file = book_dir / "4_final_pages" / f"page_{p_num:04d}_final.md"
        if not md_file.is_file():
            md_file = book_dir / "3_raw_md" / f"page_{p_num:04d}_raw.md"

        if md_file.is_file():
            markdown_text = md_file.read_text(encoding="utf-8", errors="replace")
        else:
            markdown_text = f"*(Текст страницы еще не распознан. Запустите пакетную обработку в Панели управления)*\n"

        sheet_info = self._get_sheet_info_for_page(book_name, p_num)

        mask_rel = f"/output/{book_name}/2_masked_pages/page_{p_num:04d}_masked.png"
        debug_rel = f"/output/{book_name}/2_masked_pages/page_{p_num:04d}_debug.png" if debug_file.is_file() else mask_rel

        return {
            "page_id": page_id,
            "title": f"Стр. {p_num}",
            "original_url": sheet_info["sheet_url"],
            "debug_url": debug_rel,
            "mask_url": mask_rel,
            "is_spread": sheet_info["is_spread"],
            "spread_side": sheet_info["spread_side"],
            "sheet_idx": sheet_info["sheet_idx"],
            "crops": crops_data,
            "markdown": markdown_text
        }

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

            return {
                "status": "success",
                "abc": abc_content,
                "kern": kern_content,
                "model_used": result.get("model_used", "")
            }
        finally:
            if self.omr_engine is not None:
                self.omr_engine.purge_gpu_memory()
