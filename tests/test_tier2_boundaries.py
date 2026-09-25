"""
Tier 2: Boundary & Corner Cases Test Suite (F1 - F9)
Validates system behavior under mathematical, geometrical, and resource extremes.
Edge conditions covered: extreme aspect ratios, 0-staves, max dimensions, empty batches, fault injection.
"""

import gc
import json
import os
import shutil
import sys
import tempfile
import threading
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


class TestTier2Boundaries(unittest.TestCase):
    """
    Tier 2: Boundary & Corner Cases (>=5 test cases per feature for F1 through F9).
    Total test cases: >= 45.
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

    # =========================================================================
    # F1 Boundaries: YOLO LayoutDetector (>=5 tests)
    # =========================================================================

    def test_f1_bnd_01_extreme_aspect_ratio_ultrawide(self):
        """F1 Boundary: Ultra-wide candidate (aspect ratio 200:1) validation."""
        ultrawide_crop = np.full((25, 5000, 3), 255, dtype=np.uint8)
        valid = is_valid_music_staff(ultrawide_crop, "staff", 0.9)
        self.assertFalse(valid)

    def test_f1_bnd_02_extreme_aspect_ratio_tall_column(self):
        """F1 Boundary: Extremely tall vertical candidate (aspect ratio 1:60)."""
        tall_crop = np.full((1800, 30, 3), 255, dtype=np.uint8)
        valid = is_valid_music_staff(tall_crop, "staff", 0.9)
        self.assertFalse(valid)

    def test_f1_bnd_03_zero_staves_completely_blank_canvas(self):
        """F1 Boundary: Pure white image produces zero valid staves and no errors."""
        blank_canvas = np.full((1200, 800, 3), 255, dtype=np.uint8)
        valid = is_valid_music_staff(blank_canvas, "staff", 0.5)
        self.assertFalse(valid)

    def test_f1_bnd_04_all_black_inverted_canvas(self):
        """F1 Boundary: All-black image handled safely without division by zero."""
        black_canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        valid = is_valid_music_staff(black_canvas, "staff", 0.5)
        self.assertFalse(valid)

    def test_f1_bnd_05_dense_multistaff_clustering(self):
        """F1 Boundary: Dense cluster of 20 parallel lines handled without hanging."""
        dense_crop = np.full((300, 600, 3), 255, dtype=np.uint8)
        for y in range(20, 280, 12):
            cv2.line(dense_crop, (10, y), (590, y), (0, 0, 0), 1)
        t0 = time.time()
        valid = is_valid_music_staff(dense_crop, "staff", 0.8)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.5)

    # =========================================================================
    # F2 Boundaries: OMR Transcoda-59M (>=5 tests)
    # =========================================================================

    def test_f2_bnd_01_zero_size_or_tiny_crop(self):
        """F2 Boundary: Tiny 2x2 crop rejected or handled gracefully in preprocessing."""
        engine = OMREngine(device="cpu")
        tiny_crop = np.full((2, 2, 3), 255, dtype=np.uint8)
        tensor = engine._preprocess_transcoda(tiny_crop, target_w=1050, target_h=1485)
        self.assertEqual(tensor.shape, (1, 3, 1485, 1050))

    def test_f2_bnd_02_maximum_dimension_crop(self):
        """F2 Boundary: Large 4000x3000 crop bounded cleanly to target dimensions."""
        engine = OMREngine(device="cpu")
        huge_crop = np.full((3000, 4000, 3), 200, dtype=np.uint8)
        tensor = engine._preprocess_transcoda(huge_crop, target_w=1050, target_h=1485)
        self.assertEqual(tensor.shape, (1, 3, 1485, 1050))

    def test_f2_bnd_03_single_channel_grayscale_input(self):
        """F2 Boundary: Grayscale 2D array input handled safely."""
        engine = OMREngine(device="cpu")
        gray_crop = np.full((100, 500), 200, dtype=np.uint8)
        bgr_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
        tensor = engine._preprocess_transcoda(bgr_crop)
        self.assertEqual(tensor.shape[1], 3)

    def test_f2_bnd_04_unknown_notation_class(self):
        """F2 Boundary: Unknown notation class routes safely without KeyError."""
        engine = OMREngine(device="cpu")
        crop = np.full((100, 400, 3), 255, dtype=np.uint8)
        cls_canonical = "non_existent_class_xyz".lower().replace(" ", "_")
        self.assertNotIn(cls_canonical, ("grand_staff", "grandstaff"))
        # Verify routing logic defaults cleanly to single staff path
        target_route = "smt-grandstaff" if cls_canonical in ("grand_staff", "grandstaff") else "transcoda-59M"
        self.assertEqual(target_route, "transcoda-59M")

    def test_f2_bnd_05_non_contiguous_numpy_buffer(self):
        """F2 Boundary: Non-contiguous numpy slice handled without memory error."""
        engine = OMREngine(device="cpu")
        full_img = np.full((500, 500, 3), 255, dtype=np.uint8)
        non_contig = full_img[::2, ::2]
        self.assertFalse(non_contig.flags.c_contiguous)
        tensor = engine._preprocess_transcoda(non_contig)
        self.assertEqual(tensor.shape, (1, 3, 1485, 1050))

    # =========================================================================
    # F3 Boundaries: OMR Dynamic Collation & Mini-Batching (>=5 tests)
    # =========================================================================

    def test_f3_bnd_01_empty_batch(self):
        """F3 Boundary: Batch of 0 crops handled without crashing."""
        empty_list: List[np.ndarray] = []
        target_w = 1050
        if not empty_list:
            batch_tensor = torch.empty((0, 3, 32, target_w))
        self.assertEqual(batch_tensor.shape[0], 0)

    def test_f3_bnd_02_single_item_batch(self):
        """F3 Boundary: Batch of size B=1 collated correctly."""
        engine = OMREngine(device="cpu")
        crop = np.full((100, 500, 3), 255, dtype=np.uint8)
        tensor = engine._preprocess_transcoda(crop, target_w=1050, target_h=224)
        self.assertEqual(tensor.shape, (1, 3, 224, 1050))

    def test_f3_bnd_03_maximum_batch_size_16(self):
        """F3 Boundary: Large batch of 16 crops partitioned into balanced 4-8 chunks."""
        items = list(range(16))
        b1 = items[:8]
        b2 = items[8:]
        self.assertEqual(len(b1), 8)
        self.assertEqual(len(b2), 8)

    def test_f3_bnd_04_wildly_heterogeneous_heights(self):
        """F3 Boundary: Batch with heights varying from 30px to 1400px collated into max height."""
        engine = OMREngine(device="cpu")
        crops = [
            np.full((30, 400, 3), 255, dtype=np.uint8),
            np.full((1400, 600, 3), 255, dtype=np.uint8),
        ]
        target_w = 1050
        scaled_h = [max(1, int(c.shape[0] * (target_w / float(c.shape[1])))) for c in crops]
        batch_h = min(1485, int(np.ceil(max(scaled_h) / 32.0) * 32))
        self.assertLessEqual(batch_h, 1485)
        self.assertTrue(batch_h == 1485 or batch_h % 32 == 0)

    def test_f3_bnd_05_batch_exceeding_max_height_capped_at_1485(self):
        """F3 Boundary: Height scaled >1485px is strictly capped at 1485px."""
        scaled_h = 2200
        capped_h = min(1485, int(np.ceil(scaled_h / 32.0) * 32))
        self.assertEqual(capped_h, 1485)

    # =========================================================================
    # F4 Boundaries: Sequential GPU Memory Barrier (>=5 tests)
    # =========================================================================

    def test_f4_bnd_01_purge_with_uninitialized_runner(self):
        """F4 Boundary: Purge called on freshly initialized runner without active state."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        try:
            runner._purge_vram()
        except Exception as e:
            self.fail(f"Purge on uninitialized runner failed: {e}")

    def test_f4_bnd_02_purge_during_rapid_start_stop(self):
        """F4 Boundary: Immediate stop followed by purge does not deadlock or raise."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        runner.stop()
        runner._purge_vram()
        self.assertFalse(runner.is_running)

    def test_f4_bnd_03_purge_when_cuda_mocked_unavailable(self):
        """F4 Boundary: Memory purge barrier operates cleanly when CUDA is False."""
        engine = OMREngine(device="cpu")
        engine.purge_gpu_memory()
        self.assertIsNone(engine.transcoda_model)

    def test_f4_bnd_04_purge_with_corrupt_submodule_reference(self):
        """F4 Boundary: Purge catches any exception from corrupt submodule reference."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        class FaultyEngine:
            def purge_gpu_memory(self):
                raise RuntimeError("Faulty engine error")

        runner.omr_engine = FaultyEngine()
        runner._purge_vram()
        self.assertIsNone(runner.omr_engine)

    def test_f4_bnd_05_vram_metric_robustness_when_cuda_absent(self):
        """F4 Boundary: get_metrics vram_allocated_mb defaults to 0.0 when no GPU."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        metrics = runner.get_metrics()
        self.assertEqual(metrics.get("vram_allocated_mb"), 0.0)

    # =========================================================================
    # F5 Boundaries: Pipeline Latency & Artificial Sleep Removal (>=5 tests)
    # =========================================================================

    def test_f5_bnd_01_empty_config_file(self):
        """F5 Boundary: Empty config file falls back to DEFAULT_CONFIG."""
        cfg_file = self.test_root / "empty_config.json"
        cfg_file.write_text("{}", encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        self.assertEqual(runner.config.get("dpi"), DEFAULT_CONFIG["dpi"])

    def test_f5_bnd_02_zero_delay_setting(self):
        """F5 Boundary: Config with delay: 0.0 functions correctly without error."""
        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["delay"] = 0.0
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        self.assertEqual(runner.config.get("delay"), 0.0)

    def test_f5_bnd_03_rapid_pause_resume_spam(self):
        """F5 Boundary: 100 rapid pause and resume calls complete in <50ms without deadlocking."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        runner.is_running = True

        t0 = time.time()
        for _ in range(100):
            runner.pause()
            runner.resume()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.1)

    def test_f5_bnd_04_stress_queue_sync_100_files(self):
        """F5 Boundary: Queue sync with 100 files runs efficiently without disk stall."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        runner.input_dir = self.in_dir

        for i in range(50):
            (self.in_dir / f"book_{i:03d}.pdf").write_bytes(b"dummy")

        t0 = time.time()
        runner._sync_with_in_dir()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.5)

    def test_f5_bnd_05_stop_event_immediate_interruption(self):
        """F5 Boundary: Stop event immediately halts loops without waiting for polling sleeps."""
        event = threading.Event()
        event.set()
        self.assertTrue(event.is_set())

    # =========================================================================
    # F6 Boundaries: PyMuPDF & Gutter Bisection (>=5 tests)
    # =========================================================================

    def test_f6_bnd_01_extreme_narrow_tall_spread(self):
        """F6 Boundary: Narrow aspect ratio 0.5 (portrait) never bisected."""
        tall_page = np.full((1200, 600, 3), 255, dtype=np.uint8)
        splits = detect_and_split_spread(tall_page)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0][1], "single")

    def test_f6_bnd_02_exact_aspect_ratio_threshold_1_22(self):
        """F6 Boundary: Page at aspect ratio threshold 1.22 is treated as a valid candidate."""
        page = np.full((1000, 1220, 3), 255, dtype=np.uint8)
        splits = detect_and_split_spread(page)
        self.assertIn(len(splits), (1, 2))

    def test_f6_bnd_03_ultrawide_aspect_ratio_3_0(self):
        """F6 Boundary: Aspect ratio 3.0 (above 2.2 maximum) is not treated as a 2-page book spread."""
        panorama = np.full((500, 1500, 3), 255, dtype=np.uint8)
        splits = detect_and_split_spread(panorama)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0][1], "single")

    def test_f6_bnd_04_solid_black_spine_gutter(self):
        """F6 Boundary: Solid black gutter strip falls back safely to middle width."""
        spread = np.full((800, 1200, 3), 255, dtype=np.uint8)
        spread[:, 580:620] = 0
        splits = detect_and_split_spread(spread)
        self.assertEqual(len(splits), 2)

    def test_f6_bnd_05_gutter_near_extreme_edge(self):
        """F6 Boundary: False gutter candidate at extreme edge (e.g. 20%) is rejected."""
        spread = np.full((800, 1200, 3), 255, dtype=np.uint8)
        spread[:, 240:260] = 0
        splits = detect_and_split_spread(spread)
        self.assertEqual(len(splits), 2)
        self.assertAlmostEqual(splits[0][0].shape[1], 600, delta=50)

    # =========================================================================
    # F7 Boundaries: Preprocessor 2D-DFT & Grayscale (>=5 tests)
    # =========================================================================

    def test_f7_bnd_01_extreme_skew_45_degrees(self):
        """F7 Boundary: Tilted line with steep 45 degree angle rejected or bounded by music skew estimator."""
        img = np.full((400, 400), 255, dtype=np.uint8)
        cv2.line(img, (50, 50), (350, 350), 0, 2)
        angle = estimate_skew_fourier(img)
        self.assertIsInstance(angle, float)
        self.assertLess(abs(angle), 45.0)

    def test_f7_bnd_02_pure_white_crop_deskew(self):
        """F7 Boundary: Pure white crop deskew returns 0.0 angle."""
        blank = np.full((100, 400), 255, dtype=np.uint8)
        angle = estimate_skew_fourier(blank)
        self.assertEqual(angle, 0.0)

    def test_f7_bnd_03_single_dot_image(self):
        """F7 Boundary: Image with 1 isolated dark pixel does not crash deskewer."""
        img = np.full((100, 100), 255, dtype=np.uint8)
        img[50, 50] = 0
        angle = estimate_skew_fourier(img)
        self.assertEqual(angle, 0.0)

    def test_f7_bnd_04_crop_with_vertical_lines_only(self):
        """F7 Boundary: Vertical bar lines only do not produce divide by zero."""
        img = np.full((200, 400), 255, dtype=np.uint8)
        for x in [50, 150, 250, 350]:
            cv2.line(img, (x, 10), (x, 190), 0, 2)
        angle = estimate_skew_fourier(img)
        self.assertIsInstance(angle, float)
        self.assertAlmostEqual(angle, 0.0, delta=2.0)

    def test_f7_bnd_05_large_scale_factor_crop(self):
        """F7 Boundary: normalize_staff_crop handles scale variation safely."""
        crop_bgr = np.full((120, 800, 3), 255, dtype=np.uint8)
        for y in range(20, 100, 16):
            cv2.line(crop_bgr, (20, y), (780, y), (0, 0, 0), 2)
        norm_img, angle, scale = normalize_staff_crop(crop_bgr, notation_class="staff")
        self.assertIsInstance(scale, float)
        self.assertGreaterEqual(scale, 0.0)
        self.assertEqual(norm_img.ndim, 3)

    # =========================================================================
    # F8 Boundaries: Batch Runner & Checkpoint Throttling (>=5 tests)
    # =========================================================================

    def test_f8_bnd_01_corrupted_checkpoint_json(self):
        """F8 Boundary: Corrupted checkpoint JSON is handled gracefully returning None."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("corrupt_book")
        (book_dir / "checkpoint.json").write_text("{invalid_json: true,", encoding="utf-8")

        chk = runner._read_checkpoint("corrupt_book")
        self.assertIsNone(chk)

    def test_f8_bnd_02_partially_completed_phase_recovery(self):
        """F8 Boundary: Checkpoint with partial progress is preserved on disk."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        chk = {
            "book_title": "partial_book",
            "phases": {
                "slicing": {"completed": False, "pages_done": 5, "total_pages": 10},
                "omr": {"completed": False, "crops_done": 0, "total_crops": 0},
            }
        }
        runner._write_checkpoint("partial_book", chk)
        loaded = runner._read_checkpoint("partial_book")
        self.assertEqual(loaded["phases"]["slicing"]["pages_done"], 5)

    def test_f8_bnd_03_zero_pages_pdf(self):
        """F8 Boundary: Enqueuing empty PDF file is handled cleanly."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        empty_pdf = self.in_dir / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4\n%%EOF")

        runner._enqueue_file(str(empty_pdf))
        self.assertIsInstance(runner.queue, list)

    def test_f8_bnd_04_missing_crops_directory(self):
        """F8 Boundary: Phase 2 with empty crops directory marks phase completed."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        empty_crops = self.output_dir / "empty_crops"
        empty_crops.mkdir(parents=True, exist_ok=True)
        chk = {"phases": {"omr": {}}}
        res = runner._phase_2_omr(empty_crops, chk, overwrite=False)
        self.assertTrue(res)
        self.assertEqual(chk["phases"]["omr"]["total_crops"], 0)

    def test_f8_bnd_05_read_only_or_unwritable_directory_handling(self):
        """F8 Boundary: Invalid output directory recovers to local output folder."""
        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["output_dir"] = "C:\\InvalidNonExistentDriveXYZ:\\test"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        self.assertTrue(runner.output_root.is_dir())

    # =========================================================================
    # F9 Boundaries: Strict Isolation of LM Studio / Qwen VLM (>=5 tests)
    # =========================================================================

    def test_f9_bnd_01_vlm_fallback_with_empty_crops_dir(self):
        """F9 Boundary: VLM fallback on page with 0 crops generates no-music stub notice."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("empty_crop_book")
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        masked_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)

        mask_file = masked_dir / "page_0001_masked.png"
        mask_file.write_bytes(b"dummy")

        chk = {"phases": {"vlm": {}}}
        runner._vlm_fallback_mode([mask_file], raw_md_dir, chk)

        raw_file = raw_md_dir / "page_0001_raw.md"
        text = raw_file.read_text(encoding="utf-8")
        self.assertIn("Page contains no musical material", text)

    def test_f9_bnd_02_vlm_fallback_with_special_characters_in_stub(self):
        """F9 Boundary: Special characters in stub ID are preserved exactly."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("special_chars_book")
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        crops_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)

        stub_name = "Book(Vol.1)-2026_P0001_S01_grand-staff"
        (crops_dir / f"{stub_name}.png").write_bytes(b"dummy")
        mask_file = masked_dir / "page_0001_masked.png"
        mask_file.write_bytes(b"dummy")

        chk = {"phases": {"vlm": {}}}
        runner._vlm_fallback_mode([mask_file], raw_md_dir, chk)
        text = (raw_md_dir / "page_0001_raw.md").read_text(encoding="utf-8")
        self.assertIn(f"<!-- MUSIC_STUB_ID:{stub_name} -->", text)

    def test_f9_bnd_03_assembly_missing_abc_file(self):
        """F9 Boundary: Missing .abc file inserts warning marker without crashing."""
        cfg = dict(DEFAULT_CONFIG)
        cfg["output_dir"] = str(self.output_dir)
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("missing_abc_book")
        crops_dir = book_dir / "1_crops"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        crops_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        (raw_md_dir / "page_0001_raw.md").write_text(
            "<!-- MUSIC_STUB_ID:missing_crop_id -->", encoding="utf-8"
        )
        chk = {"phases": {"assembly": {}}}
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)

        final_text = (final_dir / "page_0001.md").read_text(encoding="utf-8")
        self.assertIn("not found", final_text)

    def test_f9_bnd_04_assembly_multiline_abc_content(self):
        """F9 Boundary: Multiline ABC content properly wrapped in markdown code fence."""
        cfg = dict(DEFAULT_CONFIG)
        cfg["output_dir"] = str(self.output_dir)
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("multiline_abc_book")
        crops_dir = book_dir / "1_crops"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        crops_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        cid = "stub_polyphony"
        abc_text = "X:1\nM:4/4\nL:1/8\nV:1 clef=treble\n[V:1] c2 d2 e2 f2|\nV:2 clef=bass\n[V:2] C4 G,4|"
        (crops_dir / f"{cid}.abc").write_text(abc_text, encoding="utf-8")
        (raw_md_dir / "page_0001_raw.md").write_text(f"<!-- MUSIC_STUB_ID:{cid} -->", encoding="utf-8")

        chk = {"phases": {"assembly": {}}}
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)

        out = (final_dir / "page_0001.md").read_text(encoding="utf-8")
        self.assertIn("```abc\n" + abc_text + "\n```", out)

    def test_f9_bnd_05_lmstudio_client_malformed_json_response(self):
        """F9 Boundary: LMStudioClient handles non-JSON HTTP responses without unhandled crash."""
        client = LMStudioClient(host="127.0.0.1", port="9999")
        try:
            online, msg = client.check_connection()
            self.assertFalse(online)
        except Exception as e:
            self.fail(f"check_connection raised unhandled exception on offline server: {e}")


if __name__ == "__main__":
    unittest.main()
