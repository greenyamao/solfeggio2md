from pathlib import Path
import os
import re
import cv2
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional

from core.page_preprocessor import deskew_page, PagePreprocessor
from core.layout_detector import LayoutDetector


class PipelineWorker:
    """
    Coordinates data flow between Preprocessing, Layout Detection (YOLO),
    OMR transcription, and the GUI Workbench.
    """
    def __init__(self, output_dir: Optional[str] = None):
        self.root_dir = Path(__file__).parent.parent.resolve()
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / "test_bench" / "output"
        self.omr_dir = self.output_dir / "omr_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.omr_dir.mkdir(parents=True, exist_ok=True)
        
        self.omr_engine = None
        self.layout_detector = None
        self.preprocessor = None

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
        Scans output_dir for available processed pages.
        Returns a sorted list of page metadata summaries.
        """
        pages = []
        for item in sorted(self.output_dir.iterdir()):
            if not item.is_dir() or item.name == "omr_results":
                continue
                
            page_id = item.name
            crops_dir = item / "crops"
            num_crops = len(list(crops_dir.glob("*.png"))) if crops_dir.is_dir() else 0
            
            # Find debug and mask images
            debug_img = item / f"{page_id}_debug_boxes.png"
            mask_img = item / f"{page_id}_masked.png"
            
            # Human readable title
            title = page_id.replace("_", " ").title()
            if "P" in page_id:
                m = re.search(r"(\d+)_P(\d+)_?(left|right)?", page_id)
                if m:
                    book_id, page_num, side = m.groups()
                    side_ru = " (Левая)" if side == "left" else " (Правая)" if side == "right" else ""
                    title = f"Книга {book_id}, Стр. {int(page_num)}{side_ru}"

            pages.append({
                "page_id": page_id,
                "title": title,
                "crops_count": num_crops,
                "has_debug": debug_img.is_file(),
                "has_mask": mask_img.is_file()
            })
            
        # Put 1_P0013_left at the top if present, as it is our prime control test page
        def sort_key(p):
            pid = p["page_id"]
            if pid == "1_P0013_left":
                return "0000_1_P0013_left"
            if pid == "1_P0013_right":
                return "0001_1_P0013_right"
            return pid

        pages.sort(key=sort_key)
        return pages

    def get_page_data(self, page_id: str) -> Dict[str, Any]:
        """
        Loads full data for a given page including crops, deskewed crops,
        OMR transcriptions (ABC / Kern), and rendered markdown.
        """
        page_dir = self.output_dir / page_id
        if not page_dir.is_dir():
            raise FileNotFoundError(f"Page directory not found: {page_dir}")

        crops_dir = page_dir / "crops"
        crops_data = []

        # Find debug image
        debug_path = page_dir / f"{page_id}_debug_boxes.png"
        mask_path = page_dir / f"{page_id}_masked.png"
        original_path = page_dir / f"{page_id}_original.png"
        if not original_path.is_file() and debug_path.is_file():
            original_path = debug_path

        # Gather crops
        if crops_dir.is_dir():
            crop_files = sorted(
                [f for f in crops_dir.glob("*.png") if not f.name.endswith("_deskew.png")],
                key=lambda f: f.name
            )
            
            for idx, crop_file in enumerate(crop_files, start=1):
                crop_stem = crop_file.stem
                
                # Determine notation class from filename
                cls_name = "staff"
                if "grand_staff" in crop_stem:
                    cls_name = "grand_staff"
                elif "system" in crop_stem:
                    cls_name = "system"

                # Check or generate deskewed version
                deskew_file = crops_dir / f"{crop_stem}_deskew.png"
                skew_angle = 0.0
                
                crop_bgr = cv2.imread(str(crop_file))
                h, w = crop_bgr.shape[:2] if crop_bgr is not None else (0, 0)
                
                if crop_bgr is not None:
                    if not deskew_file.is_file():
                        deskewed_bgr, skew_angle = deskew_page(crop_bgr)
                        cv2.imwrite(str(deskew_file), deskewed_bgr)
                    else:
                        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                        from core.page_preprocessor import estimate_skew_fourier
                        skew_angle = estimate_skew_fourier(gray)

                # Look for OMR ABC and Kern files
                abc_file = self.omr_dir / f"{crop_stem}.abc"
                kern_file = self.omr_dir / f"{crop_stem}.kern"
                
                abc_content = ""
                kern_content = ""
                model_used = "Transcoda-59M" if cls_name == "staff" else "SMT-GrandStaff"
                
                if abc_file.is_file():
                    abc_content = abc_file.read_text(encoding="utf-8", errors="replace").strip()
                else:
                    # Provide placeholder if not yet transcribed
                    abc_content = f"X:{idx}\nT:{crop_stem}\nL:1/8\nM:4/4\nK:C\nV:1 treble\nz8 | z8 ||"

                if kern_file.is_file():
                    kern_content = kern_file.read_text(encoding="utf-8", errors="replace").strip()

                crops_data.append({
                    "id": f"S{idx:02d}",
                    "crop_stem": crop_stem,
                    "class": cls_name,
                    "index": idx,
                    "raw_url": f"/output/{page_id}/crops/{crop_file.name}",
                    "deskew_url": f"/output/{page_id}/crops/{deskew_file.name}" if deskew_file.is_file() else f"/output/{page_id}/crops/{crop_file.name}",
                    "skew_angle": round(skew_angle, 2),
                    "width": w,
                    "height": h,
                    "abc": abc_content,
                    "kern": kern_content,
                    "model_used": model_used
                })

        # Markdown content for Mode 4
        md_file = page_dir / f"{page_id}.md"
        if md_file.is_file():
            markdown_text = md_file.read_text(encoding="utf-8", errors="replace")
        else:
            markdown_text = self._generate_default_markdown(page_id, crops_data)
            # Save for persistence
            md_file.write_text(markdown_text, encoding="utf-8")

        # Page title
        title = page_id.replace("_", " ").title()
        m = re.search(r"(\d+)_P(\d+)_?(left|right)?", page_id)
        if m:
            book_id, page_num, side = m.groups()
            side_ru = "Левая" if side == "left" else "Правая" if side == "right" else ""
            title = f"Учебник {book_id} • Страница {int(page_num)} ({side_ru})"

        return {
            "page_id": page_id,
            "title": title,
            "debug_url": f"/output/{page_id}/{debug_path.name}" if debug_path.is_file() else "",
            "mask_url": f"/output/{page_id}/{mask_path.name}" if mask_path.is_file() else "",
            "original_url": f"/output/{page_id}/{original_path.name}" if original_path.is_file() else "",
            "crops": crops_data,
            "markdown": markdown_text,
            "hardware": self.get_hardware_status()
        }

    def _generate_default_markdown(self, page_id: str, crops: List[Dict[str, Any]]) -> str:
        """
        Generates clean, realistic Solfeggio textbook markdown with embedded ABC blocks.
        """
        m = re.search(r"(\d+)_P(\d+)_?(left|right)?", page_id)
        page_num = int(m.group(2)) if m else 1
        
        md_lines = [
            f"# УЧЕБНЫЙ КУРС СОЛЬФЕДЖИО",
            f"",
            f"## Глава IV. Двухголосные и многоголосные упражнения",
            f"",
            f"### Упражнение № {page_num}. Интонационные упражнения и слуховой анализ",
            f"",
            f"Перед сольфеджированием упражнения определите ладовую структуру, метр и ритмические особенности мелодии. Настройтесь в тональности, пропев тоническое трезвучие.",
            f""
        ]
        
        for crop in crops:
            idx = crop["index"]
            cls_name = crop["class"]
            cls_title = "Фортепианное сопровождение" if cls_name == "grand_staff" else f"Мелодический голос №{idx}"
            
            md_lines.append(f"#### {cls_title} ({cls_name})")
            md_lines.append(f"```abc")
            md_lines.append(crop["abc"])
            md_lines.append(f"```")
            md_lines.append(f"")
            md_lines.append(f"*Методическое указание*: В тактах обратите внимание на точность интонирования скачков и ритмическую пульсацию.")
            md_lines.append(f"")

        md_lines.append(f"---")
        md_lines.append(f"*Электронное издание подготовлено системой Solfeggio OCR*")
        return "\n".join(md_lines)

    def save_page_markdown(self, page_id: str, markdown_content: str) -> bool:
        """
        Persists updated markdown for a page.
        """
        page_dir = self.output_dir / page_id
        if not page_dir.is_dir():
            return False
        md_file = page_dir / f"{page_id}.md"
        md_file.write_text(markdown_content, encoding="utf-8")
        return True

    def transcribe_crop_live(self, page_id: str, crop_stem: str) -> Dict[str, Any]:
        """
        Executes real-time OMR transcription on a specific crop using OMREngine.
        Ensures strict VRAM purge immediately after.
        """
        page_dir = self.output_dir / page_id
        crops_dir = page_dir / "crops"
        crop_file = crops_dir / f"{crop_stem}.png"
        
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

        # Lazy load OMREngine to respect sequential GPU ownership
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
            
            # Save generated ABC to omr_results
            abc_content = result.get("abc", "")
            kern_content = result.get("raw_kern", "")
            
            if abc_content:
                abc_file = self.omr_dir / f"{crop_stem}.abc"
                abc_file.write_text(abc_content, encoding="utf-8")
                
            if kern_content:
                kern_file = self.omr_dir / f"{crop_stem}.kern"
                kern_file.write_text(kern_content, encoding="utf-8")

            return {
                "status": "success",
                "abc": abc_content,
                "kern": kern_content,
                "model_used": result.get("model_used", "")
            }
        finally:
            # Purge GPU memory to guarantee safety
            if self.omr_engine is not None:
                self.omr_engine.purge_gpu_memory()
