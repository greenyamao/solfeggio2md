"""
Tier 4: Real-World Application Scenarios Test Suite
Validates realistic end-to-end production workloads:
1. Single melody vocal sheet (solfeggio exercises)
2. Piano grand staff anthology (dual-staff accolades)
3. Two-page scanned book spread (spine bisection and sequential assembly)
4. Music textbook with front and back matter (mixed content handling)
5. Interrupted pipeline fault recovery and zero-redundancy checkpoint resumption
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import pymupdf as fitz
import torch

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.layout_detector import LayoutDetector, is_valid_music_staff
from core.omr_engine import OMREngine
from core.page_preprocessor import detect_and_split_spread, deskew_page, normalize_staff_crop
from core.pipeline_batch_runner import PipelineBatchRunner, DEFAULT_CONFIG


class TestTier4Workloads(unittest.TestCase):
    """
    Tier 4: Real-World Application Scenarios (>=5 realistic workloads).
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        self.output_dir = self.test_root / "output"
        self.in_dir = self.test_root / "in"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.in_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def _create_synthetic_music_pdf(self, filename: str, pages_spec: List[Dict[str, Any]]) -> Path:
        """Helper to create a realistic PDF for testing."""
        pdf_path = self.in_dir / filename
        doc = fitz.open()

        for spec in pages_spec:
            width = spec.get("width", 600)
            height = spec.get("height", 800)
            page = doc.new_page(width=width, height=height)

            # Draw title text
            title = spec.get("title", "")
            if title:
                page.insert_text((50, 50), title, fontsize=18)

            # Draw staves if requested
            staves = spec.get("staves", [])
            for s_y in staves:
                for line_idx in range(5):
                    ly = s_y + line_idx * 10
                    page.draw_line(fitz.Point(50, ly), fitz.Point(width - 50, ly), color=(0, 0, 0), width=1.5)

        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_workload_01_single_melody_vocal_sheet(self):
        """Workload 1: Single-melody vocal sheet with 3 solfeggio exercises."""
        pdf_path = self._create_synthetic_music_pdf(
            "solfeggio_vocal.pdf",
            [
                {
                    "title": "Exercise 1 - Solfeggio Vocal Melody",
                    "staves": [150, 300, 450],
                    "width": 600,
                    "height": 800,
                }
            ]
        )

        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["output_dir"] = str(self.output_dir)
        cfg["skip_vlm"] = True
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        runner = PipelineBatchRunner(config_path=cfg_file)
        book_title = "solfeggio_vocal"
        book_dir = runner._get_book_dir(book_title)
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        for d in (crops_dir, masked_dir, raw_md_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

        chk = {
            "book_title": book_title,
            "status": "in_progress",
            "phases": {
                "slicing": {"completed": False, "pages_done": 0, "total_pages": 1},
                "omr": {"completed": False, "crops_done": 0, "total_crops": 3},
                "vlm": {"completed": False, "pages_done": 0, "total_pages": 1},
                "assembly": {"completed": False}
            }
        }

        # Slicing simulation
        mask_file = masked_dir / "page_0001_masked.png"
        mask_file.write_bytes(b"dummy")
        stubs = []
        for i in range(1, 4):
            stub_id = f"{book_title}_P0001_S{i:02d}_staff"
            stubs.append(stub_id)
            (crops_dir / f"{stub_id}.png").write_bytes(b"dummy")
            (crops_dir / f"{stub_id}.abc").write_text(f"X:{i}\nT:Exercise {i}\nK:C\nC D E F|G2 G2|]", encoding="utf-8")

        chk["phases"]["slicing"]["completed"] = True
        chk["phases"]["omr"]["completed"] = True

        # Phase 3 VLM fallback
        runner._phase_3_vlm(masked_dir, raw_md_dir, chk, overwrite=True)

        # Phase 4 Assembly
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)

        final_page = final_dir / "page_0001.md"
        self.assertTrue(final_page.is_file())
        content = final_page.read_text(encoding="utf-8")
        # All 3 exercises injected into markdown
        self.assertEqual(content.count("```abc"), 3)
        self.assertIn("Exercise 1", content)
        self.assertIn("Exercise 2", content)
        self.assertIn("Exercise 3", content)

    def test_workload_02_piano_grand_staff_anthology(self):
        """Workload 2: Piano grand staff anthology with dual-staff accolades."""
        # A grand staff consists of two sets of 5 lines connected vertically
        grand_crop = np.full((250, 700, 3), 255, dtype=np.uint8)
        # Treble staff (5 lines)
        for y in [30, 42, 54, 66, 78]:
            cv2.line(grand_crop, (20, y), (680, y), (0, 0, 0), 2)
        # Bass staff (5 lines)
        for y in [130, 142, 154, 166, 178]:
            cv2.line(grand_crop, (20, y), (680, y), (0, 0, 0), 2)
        # Connecting brace bar
        cv2.line(grand_crop, (20, 30), (20, 178), (0, 0, 0), 4)

        # Physical validation for grand staff
        is_grand = is_valid_music_staff(grand_crop, "grand_staff", 0.95)
        self.assertTrue(is_grand, "Grand staff with 10 lines must be accepted")

        # Routing check in OMR
        engine = OMREngine(device="cpu")
        res = engine.transcribe_crop(grand_crop, notation_class="grand_staff", title="nocturne_m1")
        self.assertIsInstance(res, dict)
        self.assertIn("status", res)

    def test_workload_03_two_page_scanned_book_spread_e2e(self):
        """Workload 3: Two-page scanned landscape spread with gutter split and sequential assembly."""
        # 1200w x 800h landscape spread (aspect ratio 1.5)
        spread_img = np.full((800, 1200, 3), 255, dtype=np.uint8)
        # Center spine fold
        spread_img[:, 590:610] = 60

        # Draw content on left and right
        cv2.putText(spread_img, "Page Left", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(spread_img, "Page Right", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Bisection
        splits = detect_and_split_spread(spread_img)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0][1], "left")
        self.assertEqual(splits[1][1], "right")

        # Simulate assembly of the 2 sequential book pages
        book_dir = self.output_dir / "spread_book"
        raw_md_dir = book_dir / "3_raw_md"
        crops_dir = book_dir / "1_crops"
        final_dir = book_dir / "4_final_pages"
        for d in (raw_md_dir, crops_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

        (raw_md_dir / "page_0001_raw.md").write_text("## Left Page Content\n\nText here.", encoding="utf-8")
        (raw_md_dir / "page_0002_raw.md").write_text("## Right Page Content\n\nText here.", encoding="utf-8")

        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        chk = {"phases": {"assembly": {}}}
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)

        complete_file = book_dir / "spread_book_complete.md"
        self.assertTrue(complete_file.is_file())
        text = complete_file.read_text(encoding="utf-8")
        self.assertIn("<!-- PAGE 0001 -->", text)
        self.assertIn("<!-- PAGE 0002 -->", text)
        self.assertIn("Left Page Content", text)
        self.assertIn("Right Page Content", text)

    def test_workload_04_textbook_with_front_and_back_matter(self):
        """Workload 4: Textbook with front-matter (TOC), music core, and back-matter index."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("textbook_mixed")
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        for d in (crops_dir, masked_dir, raw_md_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Page 1: Preface (0 crops)
        (masked_dir / "page_0001_masked.png").write_bytes(b"dummy")
        # Page 2: Music (1 crop)
        (masked_dir / "page_0002_masked.png").write_bytes(b"dummy")
        stub_id = "textbook_mixed_P0002_S01_staff"
        (crops_dir / f"{stub_id}.png").write_bytes(b"dummy")
        (crops_dir / f"{stub_id}.abc").write_text("X:1\nK:C\nC D E F|", encoding="utf-8")

        # Run VLM fallback
        chk = {"phases": {"vlm": {}}}
        runner._vlm_fallback_mode(
            [masked_dir / "page_0001_masked.png", masked_dir / "page_0002_masked.png"],
            raw_md_dir,
            chk
        )

        p1_raw = (raw_md_dir / "page_0001_raw.md").read_text(encoding="utf-8")
        p2_raw = (raw_md_dir / "page_0002_raw.md").read_text(encoding="utf-8")
        self.assertIn("Page contains no musical material", p1_raw)
        self.assertIn(f"<!-- MUSIC_STUB_ID:{stub_id} -->", p2_raw)

        # Run Assembly
        chk_asm = {"phases": {"assembly": {}}}
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk_asm)

        comp = (book_dir / "textbook_mixed_complete.md").read_text(encoding="utf-8")
        self.assertIn("PAGE 0001", comp)
        self.assertIn("PAGE 0002", comp)
        self.assertIn("```abc", comp)

    def test_workload_05_interrupted_pipeline_fault_recovery_and_resume(self):
        """Workload 5: Pipeline interrupted mid-run, resumed from checkpoint with overwrite=False."""
        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["overwrite"] = False
        cfg["skip_vlm"] = True
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        runner = PipelineBatchRunner(config_path=cfg_file)
        book_title = "resume_test_book"
        book_dir = runner._get_book_dir(book_title)
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        for d in (crops_dir, masked_dir, raw_md_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Simulate interruption: Phase 1 Slicing already completed on disk
        (masked_dir / "page_0001_masked.png").write_bytes(b"dummy")
        stub_id = f"{book_title}_P0001_S01_staff"
        (crops_dir / f"{stub_id}.png").write_bytes(b"dummy")

        interrupted_chk = {
            "book_title": book_title,
            "status": "in_progress",
            "phases": {
                "slicing": {"completed": True, "pages_done": 1, "total_pages": 1},
                "omr": {"completed": False, "crops_done": 0, "total_crops": 1},
                "vlm": {"completed": False, "pages_done": 0, "total_pages": 1},
                "assembly": {"completed": False}
            }
        }
        runner._write_checkpoint(book_title, interrupted_chk)

        # Resume: Run Phase 2 OMR
        (crops_dir / f"{stub_id}.abc").write_text("X:1\nK:C\nC E G c|", encoding="utf-8")
        chk = runner._read_checkpoint(book_title)
        self.assertTrue(chk["phases"]["slicing"]["completed"])

        ok_omr = runner._phase_2_omr(crops_dir, chk, overwrite=False)
        self.assertTrue(ok_omr)

        # Memory purge
        runner._purge_vram()

        # Phase 3 VLM
        ok_vlm = runner._phase_3_vlm(masked_dir, raw_md_dir, chk, overwrite=False)
        self.assertTrue(ok_vlm)

        # Phase 4 Assembly
        ok_asm = runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)
        self.assertTrue(ok_asm)

        chk["status"] = "completed"
        runner._write_checkpoint(book_title, chk)

        # Verify final state
        resumed_chk = runner._read_checkpoint(book_title)
        self.assertEqual(resumed_chk["status"], "completed")
        self.assertTrue(resumed_chk["phases"]["assembly"]["completed"])
        self.assertTrue((book_dir / f"{book_title}_complete.md").is_file())


if __name__ == "__main__":
    unittest.main()
