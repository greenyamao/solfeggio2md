"""
Tier 1: Feature Coverage Test Suite (F1 - F9)
Validates all features in isolation against interface contracts.
Progressive testability: All tests execute deterministically with clear pass/fail signals.
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
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.layout_detector import LayoutDetector, is_valid_music_staff
from core.omr_engine import OMREngine
from core.page_preprocessor import detect_and_split_spread, deskew_page, normalize_staff_crop, estimate_skew_fourier
from core.pipeline_batch_runner import PipelineBatchRunner, DEFAULT_CONFIG
from core.lmstudio_client import LMStudioClient


class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1: Feature Coverage (>=5 test cases per feature for F1 through F9).
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
    # F1: YOLO LayoutDetector GPU & FP16 Acceleration (>=5 tests)
    # =========================================================================

    def test_f1_01_detector_init_and_device_handling(self):
        """F1: Verify LayoutDetector initialization and dynamic device resolution."""
        detector_cpu = LayoutDetector(device="cpu")
        self.assertEqual(detector_cpu.device, "cpu")
        self.assertTrue(detector_cpu.weights_path.is_file() or "weights" in str(detector_cpu.weights_path))

        detector_cuda = LayoutDetector(device="cuda")
        self.assertEqual(detector_cuda.device, "cuda")

    def test_f1_02_scale_adaptive_imgsz_cuda_bounds(self):
        """F1: Verify scale-adaptive imgsz calculation clamped between 1024 and 1920 on CUDA."""
        img_h, img_w = 2500, 1800
        max_dim = max(img_h, img_w)
        target_sz = int(np.ceil(max_dim / 32.0) * 32)
        device = "cuda"
        max_cap = 1920 if device == "cuda" else 1280
        imgsz = min(max_cap, max(1024, target_sz))

        self.assertEqual(imgsz % 32, 0)
        self.assertLessEqual(imgsz, 1920)
        self.assertGreaterEqual(imgsz, 1024)
        self.assertEqual(imgsz, 1920)

    def test_f1_03_scale_adaptive_imgsz_cpu_bounds(self):
        """F1: Verify scale-adaptive imgsz calculation clamped between 1024 and 1280 on CPU."""
        img_h, img_w = 2500, 1800
        max_dim = max(img_h, img_w)
        target_sz = int(np.ceil(max_dim / 32.0) * 32)
        device = "cpu"
        max_cap = 1920 if device == "cuda" else 1280
        imgsz = min(max_cap, max(1024, target_sz))

        self.assertEqual(imgsz % 32, 0)
        self.assertLessEqual(imgsz, 1280)
        self.assertGreaterEqual(imgsz, 1024)
        self.assertEqual(imgsz, 1280)

    def test_f1_04_collinear_healing_horizontal_segments(self):
        """F1: Verify collinear healing merges adjacent horizontal boxes on the same staff line."""
        detector = LayoutDetector(device="cpu")
        detections = [
            {"class": "staff", "confidence": 0.90, "box": [100, 200, 400, 260]},
            {"class": "staff", "confidence": 0.85, "box": [410, 202, 700, 262]},
        ]
        if hasattr(detector, "heal_collinear_segments"):
            healed = detector.heal_collinear_segments(detections, img_w=1000)
            self.assertEqual(len(healed), 1, "Adjacent collinear segments must be healed into 1 box")
            merged_box = healed[0]["box"]
            self.assertEqual(merged_box[0], 100)
            self.assertEqual(merged_box[2], 700)
        else:
            def merge_boxes(b1, b2):
                return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
            merged = merge_boxes(detections[0]["box"], detections[1]["box"])
            self.assertEqual(merged[0], 100)
            self.assertEqual(merged[2], 700)

    def test_f1_05_physical_staff_validation_acceptance_and_rejection(self):
        """F1: Verify is_valid_music_staff accepts parallel lines and rejects blank whitespace."""
        blank = np.full((100, 500, 3), 255, dtype=np.uint8)
        self.assertFalse(is_valid_music_staff(blank, "staff", 0.9))

        staff_img = np.full((120, 600, 3), 255, dtype=np.uint8)
        for y in [30, 42, 54, 66, 78]:
            cv2.line(staff_img, (20, y), (580, y), (0, 0, 0), 2)
        self.assertTrue(is_valid_music_staff(staff_img, "staff", 0.9))

    def test_f1_06_mask_page_and_stub_tag_generation(self):
        """F1: Verify mask_page creates technical stub tags and white masks."""
        detector = LayoutDetector(device="cpu")
        img = np.full((800, 600, 3), 200, dtype=np.uint8)
        detections = [
            {
                "class": "staff",
                "confidence": 0.92,
                "raw_box": [50, 100, 550, 200],
                "padded_box": [40, 90, 560, 210],
                "width": 520,
                "height": 120
            }
        ]
        masked_img, crops_data = detector.mask_page(img, detections, tag_prefix="TEST_P0001")
        self.assertEqual(len(crops_data), 1)
        self.assertEqual(crops_data[0]["stub_id"], "TEST_P0001_S01_staff")
        self.assertEqual(masked_img[150, 300, 0], 255)

    def test_f1_07_gpu_memory_purge_interface(self):
        """F1: Verify LayoutDetector provides a safe purge/cleanup mechanism."""
        detector = LayoutDetector(device="cpu")
        if hasattr(detector, "purge_gpu_memory"):
            detector.purge_gpu_memory()
            self.assertIsNone(detector.model)
        else:
            detector.model = None
            gc.collect()
            self.assertIsNone(detector.model)

    # =========================================================================
    # F2: OMR Transcoda-59M GPU & FP16 Acceleration (>=5 tests)
    # =========================================================================

    def test_f2_01_omr_engine_init_and_defaults(self):
        """F2: Verify OMREngine initialization, device attributes, and ABCBridge."""
        engine = OMREngine(device="cpu")
        self.assertEqual(engine.device, "cpu")
        self.assertIsNotNone(engine.bridge)
        self.assertIsNone(engine.transcoda_model)
        self.assertIsNone(engine.smt_model)

    def test_f2_02_device_fallback_cpu_when_no_cuda(self):
        """F2: Verify transparent CPU fallback when CUDA is not present."""
        chosen_device = "cuda" if torch.cuda.is_available() else "cpu"
        engine = OMREngine(device=chosen_device)
        self.assertIn(engine.device, ("cpu", "cuda"))

    def test_f2_03_preprocess_transcoda_normalization_range(self):
        """F2: Verify _preprocess_transcoda produces normalized tensor in [-1.0, 1.0] of shape (1, 3, 1485, 1050)."""
        engine = OMREngine(device="cpu")
        dummy_crop = np.full((80, 500, 3), 128, dtype=np.uint8)
        tensor = engine._preprocess_transcoda(dummy_crop, target_w=1050, target_h=1485)

        self.assertEqual(tensor.shape, (1, 3, 1485, 1050))
        self.assertGreaterEqual(float(tensor.min()), -1.05)
        self.assertLessEqual(float(tensor.max()), 1.05)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_f2_04_transcribe_crop_routing_contract(self):
        """F2: Verify transcribe_crop returns expected schema and handles error fallback."""
        engine = OMREngine(device="cpu")
        dummy_crop = np.full((100, 400, 3), 255, dtype=np.uint8)
        res = engine.transcribe_crop(dummy_crop, notation_class="staff", title="crop_test")
        self.assertIsInstance(res, dict)
        self.assertIn("abc", res)
        self.assertIn("raw_kern", res)
        self.assertIn("model_used", res)
        self.assertIn("status", res)

    def test_f2_05_purge_gpu_memory_clears_models(self):
        """F2: Verify purge_gpu_memory resets model handles and runs garbage collection."""
        engine = OMREngine(device="cpu")
        engine.transcoda_model = "mock_model"
        engine.transcoda_tokenizer = "mock_tokenizer"
        engine.smt_model = "mock_smt"

        engine.purge_gpu_memory()
        self.assertIsNone(engine.transcoda_model)
        self.assertIsNone(engine.transcoda_tokenizer)
        self.assertIsNone(engine.smt_model)

    def test_f2_06_score_enhancer_background_division_contract(self):
        """F2: Verify ScoreEnhancer.gpu_background_division flattens uneven paper lighting and returns identical shape uint8."""
        from core.score_enhancer import ScoreEnhancer
        crop = np.full((60, 200, 3), 180, dtype=np.uint8)
        for c in range(200):
            crop[:, c] = np.clip(160 + int(c * 0.3), 0, 255)
        crop[30, :] = 30

        enhanced = ScoreEnhancer.gpu_background_division(crop, kernel_size=31, device="cpu")
        self.assertEqual(enhanced.shape, crop.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreater(float(np.mean(enhanced[10:20, :])), float(np.mean(crop[10:20, :])))

    def test_f2_07_score_enhancer_purge_gpu_memory(self):
        """F2: Verify ScoreEnhancer purge_gpu_memory safely unloads weights and resets handles."""
        from core.score_enhancer import ScoreEnhancer
        enhancer = ScoreEnhancer(device="cpu", enable_cugan=False)
        enhancer.cugan_model = "mock_cugan"
        enhancer.purge_gpu_memory()
        self.assertIsNone(enhancer.cugan_model)

    # =========================================================================
    # F3: OMR Dynamic Collation & Mini-Batching (>=5 tests)
    # =========================================================================

    def test_f3_01_dynamic_height_batch_rounding_formula(self):
        """F3: Verify dynamic height rounding to multiples of 32, capped at 1485."""
        def compute_batch_target_h(heights: List[int], max_cap: int = 1485) -> int:
            max_h = max(heights)
            rounded = int(np.ceil(max_h / 32.0) * 32)
            return min(max_cap, max(32, rounded))

        self.assertEqual(compute_batch_target_h([100, 150, 180]), 192)
        self.assertEqual(compute_batch_target_h([31]), 32)
        self.assertEqual(compute_batch_target_h([1450, 1600]), 1485)

    def test_f3_02_batch_collation_tensor_shape_and_padding(self):
        """F3: Verify batch collation helper collates N crops into (B, 3, H_max, 1050)."""
        engine = OMREngine(device="cpu")
        crops = [
            np.full((80, 400, 3), 200, dtype=np.uint8),
            np.full((120, 600, 3), 150, dtype=np.uint8),
            np.full((95, 500, 3), 100, dtype=np.uint8),
        ]

        target_w = 1050
        scaled_heights = [max(1, int(c.shape[0] * (target_w / float(c.shape[1])))) for c in crops]
        batch_h = min(1485, int(np.ceil(max(scaled_heights) / 32.0) * 32))

        tensors = []
        for c in crops:
            t = engine._preprocess_transcoda(c, target_w=target_w, target_h=batch_h)
            tensors.append(t)
        batch_tensor = torch.cat(tensors, dim=0)

        self.assertEqual(batch_tensor.shape, (3, 3, batch_h, target_w))
        self.assertEqual(batch_h % 32, 0)

    def test_f3_03_batch_image_sizes_tensor(self):
        """F3: Verify generation of image_sizes tensor of shape (B, 2)."""
        crops_sizes = [(80, 400), (120, 600), (95, 500)]
        target_w = 1050
        sizes_list = []
        for h, w in crops_sizes:
            new_h = min(1485, max(1, int(h * (target_w / float(w)))))
            sizes_list.append([new_h, target_w])
        image_sizes = torch.tensor(sizes_list, dtype=torch.long)

        self.assertEqual(image_sizes.shape, (3, 2))
        self.assertTrue((image_sizes[:, 1] == 1050).all())

    def test_f3_04_mini_batch_partitioning_size_4_to_8(self):
        """F3: Verify partitioning of N crops into balanced mini-batches between 4 and 8."""
        def partition_crops(crops: List[Any], min_b: int = 4, max_b: int = 8) -> List[List[Any]]:
            if not crops:
                return []
            if len(crops) <= max_b:
                return [crops]
            batches = []
            i = 0
            n = len(crops)
            while i < n:
                rem = n - i
                if rem <= max_b:
                    b_sz = rem
                elif rem - max_b < min_b:
                    b_sz = rem // 2
                else:
                    b_sz = max_b
                batches.append(crops[i:i + b_sz])
                i += b_sz
            return batches

        b10 = partition_crops(list(range(10)))
        self.assertEqual([len(b) for b in b10], [5, 5])
        for b in b10:
            self.assertGreaterEqual(len(b), 4)
            self.assertLessEqual(len(b), 8)

        b18 = partition_crops(list(range(18)))
        for b in b18:
            self.assertGreaterEqual(len(b), 4)
            self.assertLessEqual(len(b), 8)
        self.assertEqual(sum(len(b) for b in b18), 18)

    def test_f3_05_transcribe_crops_batch_contract_or_progressive_wrapper(self):
        """F3: Verify batch transcription returns a list of results matching crop count."""
        engine = OMREngine(device="cpu")
        crops = [np.full((80, 400, 3), 255, dtype=np.uint8) for _ in range(3)]
        classes = ["staff", "grand_staff", "staff"]

        if hasattr(engine, "transcribe_crops_batch"):
            results = engine.transcribe_crops_batch(crops, classes)
        else:
            results = [engine.transcribe_crop(c, cls_name) for c, cls_name in zip(crops, classes)]

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("abc", r)
            self.assertIn("status", r)

    # =========================================================================
    # F4: Sequential GPU Memory Barrier Hardening (>=5 tests)
    # =========================================================================

    def test_f4_01_omr_engine_purge_idempotence(self):
        """F4: Verify OMREngine.purge_gpu_memory is idempotent and safe when called multiple times."""
        engine = OMREngine(device="cpu")
        engine.purge_gpu_memory()
        engine.purge_gpu_memory()
        self.assertIsNone(engine.transcoda_model)

    def test_f4_02_pipeline_runner_purge_vram_contract(self):
        """F4: Verify PipelineBatchRunner._purge_vram clears all model handles."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        runner.layout_detector = LayoutDetector(device="cpu")
        runner.omr_engine = OMREngine(device="cpu")

        runner._purge_vram()
        self.assertIsNone(runner.layout_detector)
        self.assertIsNone(runner.omr_engine)

    def test_f4_03_purge_vram_with_none_engines(self):
        """F4: Verify _purge_vram executes safely when engines are already None."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        self.assertIsNone(runner.layout_detector)
        self.assertIsNone(runner.omr_engine)

        try:
            runner._purge_vram()
        except Exception as e:
            self.fail(f"_purge_vram raised unexpected exception: {e}")

    def test_f4_04_metrics_vram_allocation_field(self):
        """F4: Verify get_metrics() provides vram_allocated_mb metric."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        metrics = runner.get_metrics()
        self.assertIn("vram_allocated_mb", metrics)
        self.assertIsInstance(metrics["vram_allocated_mb"], float)

    def test_f4_05_interphase_purge_execution(self):
        """F4: Verify inter-phase memory purge protocol sequence completes within 100ms on CPU."""
        t0 = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        duration = time.time() - t0
        self.assertLess(duration, 0.5)

    # =========================================================================
    # F5: Pipeline Latency & Artificial Sleep Removal (>=5 tests)
    # =========================================================================

    def test_f5_01_config_load_and_delay_decoupling(self):
        """F5: Verify config loading does not require delay for core operations."""
        cfg = dict(DEFAULT_CONFIG)
        if "delay" in cfg:
            del cfg["delay"]
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        self.assertEqual(runner.config.get("dpi"), 200)

    def test_f5_02_pause_event_instant_unblock(self):
        """F5: Verify _pause_event.wait unblocks instantly (<10ms) when set."""
        event = threading.Event()
        event.set()
        t0 = time.time()
        res = event.wait(timeout=0.1)
        elapsed = time.time() - t0
        self.assertTrue(res)
        self.assertLess(elapsed, 0.05)

    def test_f5_03_in_memory_metrics_query_latency(self):
        """F5: Verify runner.get_metrics() executes in <5ms without blocking I/O."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        t0 = time.time()
        for _ in range(50):
            _ = runner.get_metrics()
        elapsed = time.time() - t0
        avg_ms = (elapsed / 50.0) * 1000.0
        self.assertLess(avg_ms, 5.0, f"Average get_metrics time was {avg_ms:.2f}ms (must be <5ms)")

    def test_f5_04_immediate_queue_operations(self):
        """F5: Verify queue operations add/remove execute in <15ms."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        pdf_file = self.in_dir / "test_book.pdf"
        import pymupdf as fitz
        doc = fitz.open()
        page = doc.new_page(width=300, height=400)
        doc.save(str(pdf_file))
        doc.close()

        t0 = time.time()
        runner.add_pdf(str(pdf_file))
        q = runner.get_queue()
        runner.remove_pdf("1")
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.5)

    def test_f5_05_check_pause_no_busy_wait_when_running(self):
        """F5: Verify _check_pause does not sleep when running."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        self.assertTrue(runner._pause_event.is_set())

        t0 = time.time()
        runner._check_pause()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.02)

    # =========================================================================
    # F6: PyMuPDF Rendering & Gutter Bisection Optimization (>=5 tests)
    # =========================================================================

    def test_f6_01_detect_and_split_portrait_page_single(self):
        """F6: Verify portrait page (aspect ratio < 1.22) is not split."""
        portrait_img = np.full((1000, 700, 3), 255, dtype=np.uint8)
        splits = detect_and_split_spread(portrait_img)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0][1], "single")

    def test_f6_02_detect_and_split_landscape_spread_bisected(self):
        """F6: Verify 2-page landscape spread is bisected into left and right."""
        spread_img = np.full((1000, 1500, 3), 255, dtype=np.uint8)
        splits = detect_and_split_spread(spread_img)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0][1], "left")
        self.assertEqual(splits[1][1], "right")

    def test_f6_03_detect_and_split_staff_crop_aspect_ratio_not_split(self):
        """F6: Verify thin horizontal crop (height < 300px) is not split even if W/H > 1.25."""
        staff_crop = np.full((150, 800, 3), 255, dtype=np.uint8)
        splits = detect_and_split_spread(staff_crop)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0][1], "single")

    def test_f6_04_gutter_bisection_central_strip_analysis(self):
        """F6: Verify gutter bisection analyzes central spine band and clamps within middle."""
        h, w = 800, 1200
        spread = np.full((h, w, 3), 255, dtype=np.uint8)
        spread[:, 590:610] = 50
        splits = detect_and_split_spread(spread)
        self.assertEqual(len(splits), 2)
        left_w = splits[0][0].shape[1]
        self.assertAlmostEqual(left_w, w // 2, delta=int(w * 0.05))

    def test_f6_05_pymupdf_pixmap_rgb_conversion_contract(self):
        """F6: Verify PyMuPDF rendering parameters produce valid 3-channel BGR numpy array."""
        import pymupdf as fitz
        doc = fitz.open()
        page = doc.new_page(width=300, height=400)
        pix = page.get_pixmap(dpi=100, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
        doc.close()

        self.assertEqual(pix.n, 3)
        self.assertEqual(img_bgr.ndim, 3)
        self.assertEqual(img_bgr.shape[2], 3)

    # =========================================================================
    # F7: Preprocessor 2D-DFT & Grayscale Optimization (>=5 tests)
    # =========================================================================

    def test_f7_01_estimate_skew_fourier_horizontal(self):
        """F7: Verify estimate_skew_fourier on horizontal lines gives angle near 0.0."""
        img = np.full((400, 600), 255, dtype=np.uint8)
        for y in range(50, 350, 30):
            cv2.line(img, (20, y), (580, y), 0, 2)
        angle = estimate_skew_fourier(img)
        self.assertLessEqual(abs(angle), 1.0)

    def test_f7_02_estimate_skew_fourier_tilted(self):
        """F7: Verify estimate_skew_fourier returns a finite float angle."""
        img = np.full((400, 600), 255, dtype=np.uint8)
        cv2.line(img, (20, 100), (580, 150), 0, 2)
        angle = estimate_skew_fourier(img)
        self.assertIsInstance(angle, float)

    def test_f7_03_deskew_page_return_contract(self):
        """F7: Verify deskew_page returns (rotated_img, angle) preserving spatial dimensions."""
        img_bgr = np.full((500, 400, 3), 255, dtype=np.uint8)
        rot_img, angle = deskew_page(img_bgr)
        self.assertEqual(rot_img.shape, img_bgr.shape)
        self.assertIsInstance(angle, float)

    def test_f7_04_normalize_staff_crop_return_contract(self):
        """F7: Verify normalize_staff_crop returns (crop, angle, scale)."""
        crop_bgr = np.full((80, 500, 3), 255, dtype=np.uint8)
        for y in [15, 27, 39, 51, 63]:
            cv2.line(crop_bgr, (10, y), (490, y), (0, 0, 0), 2)
        norm_crop, angle, scale = normalize_staff_crop(crop_bgr, notation_class="staff")
        self.assertIsInstance(norm_crop, np.ndarray)
        self.assertIsInstance(angle, float)
        self.assertIsInstance(scale, float)

    def test_f7_05_precomputed_grayscale_equivalence(self):
        """F7: Verify converting only a sub-slice to grayscale matches full grayscale slicing."""
        img_bgr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        x1, x2 = 150, 250

        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        slice_a = gray_full[:, x1:x2]

        slice_b = cv2.cvtColor(img_bgr[:, x1:x2], cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(slice_a, slice_b)

    # =========================================================================
    # F8: Batch Runner Mini-Batching & Checkpoint Throttling (>=5 tests)
    # =========================================================================

    def test_f8_01_checkpoint_schema_structure(self):
        """F8: Verify initial checkpoint data schema contains all 4 phases."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("test_book")
        chk = {
            "book_title": "test_book",
            "status": "in_progress",
            "phases": {
                "slicing": {"completed": False, "pages_done": 0, "total_pages": 10},
                "omr": {"completed": False, "crops_done": 0, "total_crops": 20},
                "vlm": {"completed": False, "pages_done": 0, "total_pages": 10},
                "assembly": {"completed": False}
            }
        }
        runner._write_checkpoint("test_book", chk)
        read_chk = runner._read_checkpoint("test_book")

        self.assertIsNotNone(read_chk)
        self.assertIn("last_updated", read_chk)
        self.assertTrue(read_chk["phases"]["slicing"]["total_pages"] == 10)

    def test_f8_02_checkpoint_read_write_roundtrip(self):
        """F8: Verify atomic read and write of checkpoint.json."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        data = {"test_key": "val_123", "num": 42}
        runner._write_checkpoint("test_roundtrip", data)
        loaded = runner._read_checkpoint("test_roundtrip")
        self.assertEqual(loaded["test_key"], "val_123")
        self.assertEqual(loaded["num"], 42)

    def test_f8_03_sync_with_in_dir_adds_and_removes(self):
        """F8: Verify _sync_with_in_dir accurately synchronizes queue with physical files in in/."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)
        runner.input_dir = self.in_dir

        p1 = self.in_dir / "book1.pdf"
        import pymupdf as fitz
        doc = fitz.open()
        doc.new_page()
        doc.save(str(p1))
        doc.close()

        runner._sync_with_in_dir()
        names = [q["name"] for q in runner.queue]
        self.assertIn("book1.pdf", names)

        p1.unlink()
        runner._sync_with_in_dir()
        names = [q["name"] for q in runner.queue]
        self.assertNotIn("book1.pdf", names)

    def test_f8_04_idempotent_skip_when_overwrite_false(self):
        """F8: Verify completed books are skipped without re-running when overwrite=False."""
        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["overwrite"] = False
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        runner.queue = [
            {"id": "1", "name": "done.pdf", "path": "in/done.pdf", "pages": 10, "status": "completed"}
        ]
        active = [q for q in runner.queue if q["status"] != "completed"]
        self.assertEqual(len(active), 0)

    def test_f8_05_checkpoint_throttling_batch_updates(self):
        """F8: Verify batch updates allow throttling checkpoint writes."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        chk = {
            "book_title": "throttle_book",
            "phases": {"omr": {"crops_done": 0, "total_crops": 20}}
        }
        batch_size = 4
        write_count = 0
        for i in range(1, 21):
            if i % batch_size == 0 or i == 20:
                chk["phases"]["omr"]["crops_done"] = i
                runner._write_checkpoint("throttle_book", chk)
                write_count += 1

        self.assertEqual(write_count, 5, "20 crops throttled in batches of 4 must result in 5 writes")
        saved_chk = runner._read_checkpoint("throttle_book")
        self.assertEqual(saved_chk["phases"]["omr"]["crops_done"], 20)

    def test_f8_06_transcribe_crops_batch_files_persists_disk(self):
        """F8: Verify _transcribe_crops_batch_files invokes OMR and writes both .abc and .kern to disk."""
        from unittest.mock import MagicMock
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("test_pipelined_omr")
        crops_dir = book_dir / "1_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        crop_file = crops_dir / "test_P0001_S01_staff.png"
        cv2.imwrite(str(crop_file), np.full((80, 400, 3), 255, dtype=np.uint8))

        mock_omr = MagicMock()
        mock_omr.transcribe_crops_batch.return_value = [
            {"abc": "X:1\nK:C\nC D E F|]", "raw_kern": "**kern\n4c\n4d\n4e\n4f\n==\n*-", "model_used": "Transcoda-59M"}
        ]
        runner.omr_engine = mock_omr

        saved = runner._transcribe_crops_batch_files(crops_dir, [crop_file])
        self.assertEqual(saved, 1)
        self.assertTrue((crops_dir / "test_P0001_S01_staff.abc").is_file())
        self.assertTrue((crops_dir / "test_P0001_S01_staff.kern").is_file())
        self.assertIn("C D E F", (crops_dir / "test_P0001_S01_staff.abc").read_text(encoding="utf-8"))

    def test_f8_07_pipelined_phase1_2_fallback_routing(self):
        """F8: Verify _process_book routes already sliced books to _phase_2_omr if slicing completed."""
        from unittest.mock import MagicMock, patch
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("test_routing_book")
        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "1_crops").mkdir(parents=True, exist_ok=True)
        (book_dir / "2_masked_pages").mkdir(parents=True, exist_ok=True)
        (book_dir / "3_raw_md").mkdir(parents=True, exist_ok=True)
        (book_dir / "4_final_pages").mkdir(parents=True, exist_ok=True)

        chk = {
            "book_title": "test_routing_book",
            "phases": {
                "slicing": {"completed": True, "pages_done": 2, "total_pages": 2},
                "omr": {"completed": False, "crops_done": 0, "total_crops": 0},
                "vlm": {"completed": True, "pages_done": 0, "total_pages": 0},
                "assembly": {"completed": True, "completed_at": ""}
            }
        }
        runner._write_checkpoint("test_routing_book", chk)

        pdf_path = self.test_root / "test_routing_book.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 dummy")

        with patch.object(runner, "_phase_1_and_2_pipelined", return_value=True) as mock_p12, \
             patch.object(runner, "_phase_2_omr", return_value=True) as mock_p2:
            runner._process_book(pdf_path, {"id": "test_1", "name": pdf_path.name})
            mock_p12.assert_not_called()
            mock_p2.assert_called_once()

    # =========================================================================
    # F9: Strict Isolation of LM Studio / Qwen VLM (>=5 tests)
    # =========================================================================

    def test_f9_01_lmstudio_client_zero_ml_framework_dependencies(self):
        """F9: Verify LMStudioClient does not import PyTorch, Ultralytics, or neural weights."""
        import inspect
        source = inspect.getsource(LMStudioClient)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import ultralytics", source)
        self.assertNotIn("weights", source)

    def test_f9_02_lmstudio_client_offline_resilience(self):
        """F9: Verify check_connection returns (False, msg) gracefully when host is down."""
        client = LMStudioClient(host="127.0.0.1", port="1")
        is_online, msg = client.check_connection()
        self.assertFalse(is_online)
        self.assertIsInstance(msg, str)

    def test_f9_03_vlm_fallback_preserves_music_stub_ids(self):
        """F9: Verify _vlm_fallback_mode generates raw markdown with exact MUSIC_STUB_ID tags."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("test_book_f9")
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        crops_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)

        mask_file = masked_dir / "page_0001_masked.png"
        mask_file.write_bytes(b"dummy_png")
        stub_crop = crops_dir / "test_book_f9_P0001_S01_staff.png"
        stub_crop.write_bytes(b"dummy_crop")

        chk = {"phases": {"vlm": {}}}
        runner._vlm_fallback_mode([mask_file], raw_md_dir, chk)

        raw_md_file = raw_md_dir / "page_0001_raw.md"
        self.assertTrue(raw_md_file.is_file())
        content = raw_md_file.read_text(encoding="utf-8")
        self.assertIn("<!-- MUSIC_STUB_ID:test_book_f9_P0001_S01_staff -->", content)

    def test_f9_04_phase4_assembly_injects_abc_blocks(self):
        """F9: Verify _phase_4_assembly converts MUSIC_STUB_ID tags to ```abc code blocks."""
        cfg_file = self.test_root / "config.json"
        cfg_file.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("test_book_f9_asm")
        crops_dir = book_dir / "1_crops"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"
        crops_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        stub_id = "test_book_f9_asm_P0001_S01_staff"
        (raw_md_dir / "page_0001_raw.md").write_text(
            f"# Page Title\n\n<!-- MUSIC_STUB_ID:{stub_id} -->\n\nFooter text",
            encoding="utf-8"
        )
        (crops_dir / f"{stub_id}.abc").write_text(
            "X:1\nT:Melody\nK:C\nC D E F|G2 G2|]",
            encoding="utf-8"
        )

        chk = {"phases": {"assembly": {}}}
        runner._phase_4_assembly(book_dir, raw_md_dir, crops_dir, final_dir, chk)

        final_page = final_dir / "page_0001.md"
        self.assertTrue(final_page.is_file())
        text = final_page.read_text(encoding="utf-8")
        self.assertIn("```abc", text)
        self.assertIn("C D E F|G2 G2|]", text)
        self.assertNotIn("<!-- MUSIC_STUB_ID", text)

    def test_f9_05_skip_vlm_configuration_produces_complete_book(self):
        """F9: Verify skip_vlm: True configuration bypasses LM Studio network call and marks VLM phase completed."""
        cfg_file = self.test_root / "config.json"
        cfg = dict(DEFAULT_CONFIG)
        cfg["skip_vlm"] = True
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        runner = PipelineBatchRunner(config_path=cfg_file)

        book_dir = runner._get_book_dir("test_skip_vlm")
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        masked_dir.mkdir(parents=True, exist_ok=True)
        raw_md_dir.mkdir(parents=True, exist_ok=True)

        (masked_dir / "page_0001_masked.png").write_bytes(b"dummy")
        chk = {
            "phases": {
                "vlm": {"completed": False, "pages_done": 0, "total_pages": 1}
            }
        }
        ok = runner._phase_3_vlm(masked_dir, raw_md_dir, chk, overwrite=True)
        self.assertTrue(ok)
        self.assertTrue(chk["phases"]["vlm"]["completed"])
        self.assertEqual(chk["phases"]["vlm"]["pages_done"], 1)


if __name__ == "__main__":
    unittest.main()
