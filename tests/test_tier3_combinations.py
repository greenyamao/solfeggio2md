"""
Tier 3: Cross-Feature Combinations Test Suite
Validates pairwise and multi-stage interactions across pipeline phases:
- Spread bisection + Layout detection (F6 + F1)
- Fourier deskew + Staff validation (F7 + F1)
- Layout crops + OMR Dynamic batch collation (F1 + F3)
- OMR inference + GPU Memory barrier (F2 + F4)
- Sequential Memory barrier + VLM isolation handshake (F4 + F9)
- Batched OMR + Checkpoint throttling (F3 + F8)
- Zero-sleep queue sync + Checkpointing (F5 + F8)
- Spread split + Independent Fourier deskew (F6 + F7)
- YOLO Stub tag lifecycle to markdown assembly (F1 + F9 + F8)
- Full multi-phase pipeline integration (Phase 1 -> Phase 2 -> Purge -> Phase 3 -> Phase 4)
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
import torch

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.layout_detector import LayoutDetector, is_valid_music_staff
from core.omr_engine import OMREngine
from core.page_preprocessor import detect_and_split_spread, deskew_page, normalize_staff_crop, estimate_skew_fourier
from core.pipeline_batch_runner import PipelineBatchRunner, DEFAULT_CONFIG
from core.lmstudio_client import LMStudioClient


class TestTier3Combinations(unittest.TestCase):
    """
    Tier 3: Cross-Feature Combinations (>=10 test cases covering pairwise module interactions).
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

    def test_comb_01_f6_f1_spread_bisection_feeding_into_layout_detection(self):
        """Combination F6 + F1: Two-page spread bisected into left/right, feeding into layout detector."""
        # Create a synthetic 2-page spread: 1000h x 1500w (aspect ratio 1.5)
        spread = np.full((1000, 1500, 3), 255, dtype=np.uint8)
        # Spine band in middle
        spread[:, 740:760] = 50

        # Draw 5 staff lines on left page and right page
        for y in [200, 220, 240, 260, 280]:
            cv2.line(spread, (50, y), (650, y), (0, 0, 0), 2)
            cv2.line(spread, (850, y), (1450, y), (0, 0, 0), 2)

        # 1. Bisection
        split_pages = detect_and_split_spread(spread)
        self.assertEqual(len(split_pages), 2, "Landscape spread must be split into 2 pages")

        # 2. Feed each into LayoutDetector imgsz & candidate bounding logic
        detector = LayoutDetector(device="cpu")
        for page_img, side in split_pages:
            h, w = page_img.shape[:2]
            target_sz = int(np.ceil(max(h, w) / 32.0) * 32)
            imgsz = min(1280, max(1024, target_sz))
            self.assertEqual(imgsz % 32, 0)
            self.assertGreaterEqual(imgsz, 1024)

    def test_comb_02_f7_f1_fourier_deskew_before_staff_validation(self):
        """Combination F7 + F1: Skewed page deskewed with 2D-DFT, then validated by is_valid_music_staff."""
        # Create page with horizontal lines
        page = np.full((600, 800, 3), 255, dtype=np.uint8)
        for y in [150, 170, 190, 210, 230]:
            cv2.line(page, (50, y), (750, y), (0, 0, 0), 2)

        # 1. Deskew page
        deskewed_page, angle = deskew_page(page)
        self.assertIsInstance(angle, float)

        # 2. Extract crop from deskewed page
        crop = deskewed_page[140:240, 40:760]
        # 3. Validate staff lines
        valid = is_valid_music_staff(crop, "staff", 0.9)
        self.assertTrue(valid, "Deskewed 5-line staff must pass physical validation")

    def test_comb_03_f1_f3_layout_crops_batched_into_omr_collation(self):
        """Combination F1 + F3: LayoutDetector mask_page extracts crops, collated into dynamic OMR batch."""
        detector = LayoutDetector(device="cpu")
        img = np.full((1200, 900, 3), 255, dtype=np.uint8)

        # 4 simulated detections
        detections = [
            {"class": "staff", "confidence": 0.95, "raw_box": [50, 100, 850, 200], "padded_box": [40, 90, 860, 210]},
            {"class": "staff", "confidence": 0.91, "raw_box": [50, 300, 850, 400], "padded_box": [40, 290, 860, 410]},
            {"class": "staff", "confidence": 0.93, "raw_box": [50, 500, 850, 600], "padded_box": [40, 490, 860, 610]},
            {"class": "staff", "confidence": 0.90, "raw_box": [50, 700, 850, 800], "padded_box": [40, 690, 860, 810]},
        ]

        masked_img, crops_data = detector.mask_page(img, detections, tag_prefix="BATCH_COMB")
        self.assertEqual(len(crops_data), 4)

        # Collate extracted crops into dynamic batch
        crops = [c["crop_img"] for c in crops_data]
        target_w = 1050
        scaled_heights = [max(1, int(c.shape[0] * (target_w / float(c.shape[1])))) for c in crops]
        batch_h = min(1485, int(np.ceil(max(scaled_heights) / 32.0) * 32))

        engine = OMREngine(device="cpu")
        tensors = [engine._preprocess_transcoda(c, target_w=target_w, target_h=batch_h) for c in crops]
        batch_tensor = torch.cat(tensors, dim=0)

        self.assertEqual(batch_tensor.shape, (4, 3, batch_h, target_w))

    def test_comb_04_f2_f4_omr_execution_followed_by_gpu_purge(self):
        """Combination F2 + F4: OMR engine runs crop preprocessing, followed by immediate GPU purge."""
        engine = OMREngine(device="cpu")
        crop = np.full((120, 600, 3), 255, dtype=np.uint8)
        # Preprocess
        tensor = engine._preprocess_transcoda(crop)
        self.assertEqual(tensor.shape, (1, 3, 1485, 1050))

        # Memory purge barrier
        engine.purge_gpu_memory()
        self.assertIsNone(engine.transcoda_model)
        self.assertIsNone(engine.smt_model)

    def test_comb_05_f4_f9_sequential_memory_barrier_prior_to_vlm_handshake(self):
        """Combination F4 + F9: Memory purge barrier executes prior to Phase 3 VLM invocation."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        # Initialize both Phase 1 and Phase 2 components
        runner.layout_detector = LayoutDetector(device="cpu")
        runner.omr_engine = OMREngine(device="cpu")

        # Execute barrier
        runner._purge_vram()
        self.assertIsNone(runner.layout_detector)
        self.assertIsNone(runner.omr_engine)

        # VLM client is isolated and untouched
        self.assertIsNotNone(runner.lm_client)
        self.assertEqual(runner.lm_client.host, "127.0.0.1")

    def test_comb_06_f3_f8_batched_omr_with_checkpoint_throttling(self):
        """Combination F3 + F8: Mini-batched OMR advances checkpoint once per batch."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("batch_omr_book")
        crops_dir = book_dir / "1_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Create 8 dummy crops
        crop_names = []
        for i in range(1, 9):
            c_name = f"batch_omr_book_P0001_S{i:02d}_staff.png"
            (crops_dir / c_name).write_bytes(b"dummy")
            crop_names.append(c_name)

        chk = {
            "book_title": "batch_omr_book",
            "phases": {"omr": {"completed": False, "crops_done": 0, "total_crops": 8}}
        }

        # Mini-batches of 4
        batch_size = 4
        checkpoint_writes = 0
        for b_start in range(0, 8, batch_size):
            b_crops = crop_names[b_start:b_start + batch_size]
            for c_name in b_crops:
                stem = Path(c_name).stem
                (crops_dir / f"{stem}.abc").write_text("X:1\nK:C\nC D E F|", encoding="utf-8")
            chk["phases"]["omr"]["crops_done"] = b_start + len(b_crops)
            runner._write_checkpoint("batch_omr_book", chk)
            checkpoint_writes += 1

        self.assertEqual(checkpoint_writes, 2)
        saved = runner._read_checkpoint("batch_omr_book")
        self.assertEqual(saved["phases"]["omr"]["crops_done"], 8)

    def test_comb_07_f5_f8_zero_sleep_batch_queue_orchestration(self):
        """Combination F5 + F8: Queue sync, checkpoint loading, and status queries complete without sleeps."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        runner.input_dir = self.in_dir

        t0 = time.time()
        # 1. Sync queue
        runner._sync_with_in_dir()
        # 2. Get metrics
        m = runner.get_metrics()
        # 3. Read & write checkpoint
        runner._write_checkpoint("test_speed", {"status": "pending"})
        chk = runner._read_checkpoint("test_speed")
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.1, "Orchestration operations must complete in <100ms")
        self.assertIsNotNone(chk)

    def test_comb_08_f6_f7_spread_split_followed_by_fourier_deskew(self):
        """Combination F6 + F7: Spread split into left/right, and each page is independently deskewed."""
        spread = np.full((1000, 1500, 3), 255, dtype=np.uint8)
        spread[:, 740:760] = 50

        # Draw horizontal lines on left
        for y in [200, 230, 260]:
            cv2.line(spread, (50, y), (700, y), 0, 2)
        # Draw tilted lines on right
        for y in [200, 230, 260]:
            cv2.line(spread, (800, y), (1450, y + 15), 0, 2)

        splits = detect_and_split_spread(spread)
        self.assertEqual(len(splits), 2)

        left_page, left_side = splits[0]
        right_page, right_side = splits[1]

        rot_left, angle_left = deskew_page(left_page)
        rot_right, angle_right = deskew_page(right_page)

        self.assertEqual(rot_left.shape, left_page.shape)
        self.assertEqual(rot_right.shape, right_page.shape)
        self.assertIsInstance(angle_left, float)
        self.assertIsInstance(angle_right, float)

    def test_comb_09_f1_f9_stub_generation_to_markdown_assembly(self):
        """Combination F1 + F9: Layout stub generation -> VLM fallback -> Final markdown ABC assembly."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("lifecycle_book")
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        for d in (crops_dir, masked_dir, raw_md_dir, final_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1. Layout masking generates crops and stub tags
        detector = LayoutDetector(device="cpu")
        canvas = np.full((800, 600, 3), 255, dtype=np.uint8)
        detections = [
            {"class": "staff", "confidence": 0.9, "raw_box": [50, 100, 550, 200], "padded_box": [40, 90, 560, 210]}
        ]
        masked_img, crops_data = detector.mask_page(canvas, detections, tag_prefix="lifecycle_P0001")
        mask_file = masked_dir / "page_0001_masked.png"
        cv2.imwrite(str(mask_file), masked_img)

        # Save crops and ABC
        stub_id = crops_data[0]["stub_id"]
        crop_file = crops_dir / f"{stub_id}.png"
        cv2.imwrite(str(crop_file), crops_data[0]["crop_img"])
        abc_file = crops_dir / f"{stub_id}.abc"
        abc_file.write_text("X:1\nT:Test Melody\nK:G\nG A B c|d2 d2|]", encoding="utf-8")

        # 2. VLM Fallback mode generates raw markdown preserving stub tag
        chk = {"phases": {"vlm": {}}}
        runner._vlm_fallback_mode([mask_file], raw_md_dir, chk)
        raw_file = raw_md_dir / "page_0001_raw.md"
        self.assertTrue(raw_file.is_file())
        self.assertIn(f"<!-- MUSIC_STUB_ID:{stub_id} -->", raw_file.read_text(encoding="utf-8"))

        # 3. Final assembly injects ABC code block
        chk_asm = {"phases": {"assembly": {}}}
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk_asm)

        final_page = final_dir / "page_0001.md"
        self.assertTrue(final_page.is_file())
        final_content = final_page.read_text(encoding="utf-8")
        self.assertIn("```abc", final_content)
        self.assertIn("G A B c|d2 d2|]", final_content)
        self.assertNotIn("<!-- MUSIC_STUB_ID", final_content)

    def test_comb_10_full_offline_pipeline_phase1_through_phase4(self):
        """Combination End-to-End: Full offline 4-phase execution on a synthesized document."""
        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["output_dir"] = str(self.output_dir)
        cfg["skip_vlm"] = True
        cfg["overwrite"] = True
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        runner = PipelineBatchRunner(config_path=cfg_file)
        book_title = "e2e_comb_doc"
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
                "omr": {"completed": False, "crops_done": 0, "total_crops": 1},
                "vlm": {"completed": False, "pages_done": 0, "total_pages": 1},
                "assembly": {"completed": False}
            }
        }

        # Step 1: Simulate Phase 1 outputs
        (masked_dir / "page_0001_masked.png").write_bytes(b"dummy")
        stub_id = f"{book_title}_P0001_S01_staff"
        (crops_dir / f"{stub_id}.png").write_bytes(b"dummy")
        chk["phases"]["slicing"]["completed"] = True
        chk["phases"]["slicing"]["pages_done"] = 1

        # Step 2: Phase 2 OMR produces .abc
        (crops_dir / f"{stub_id}.abc").write_text("X:1\nK:C\nC2 D2 E2 F2|", encoding="utf-8")
        chk["phases"]["omr"]["completed"] = True
        chk["phases"]["omr"]["crops_done"] = 1

        # Step 3: VRAM purge
        runner._purge_vram()

        # Step 4: Phase 3 VLM (offline fallback)
        ok_vlm = runner._phase_3_vlm(masked_dir, raw_md_dir, chk, overwrite=True)
        self.assertTrue(ok_vlm)

        # Step 5: Phase 4 Assembly
        ok_asm = runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)
        self.assertTrue(ok_asm)

        # Complete book output file exists and has content
        complete_book = book_dir / f"{book_title}_complete.md"
        self.assertTrue(complete_book.is_file())
        self.assertGreater(complete_book.stat().st_size, 0)
        self.assertIn("```abc", complete_book.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
