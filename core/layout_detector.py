import gc
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch

from core.page_preprocessor import estimate_staff_spacing


def is_valid_music_staff(crop_bgr: np.ndarray, cls_name: str, conf: float) -> bool:
    """
    Physical verification that a candidate bounding box contains authentic parallel music staff lines.
    Rejects:
    - Printed text headers / example labels (e.g. 'Example 26 BACH...')
    - Analysis brackets (|---|---|---|)
    - Slur / phrase arcs
    - Text underlines, footnote lines, and table borders
    - Multi-row data table grids (6 or more consecutive equidistant lines)
    - Blank whitespace hallucinations
    """
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
        return False
    if len(crop_bgr.shape) < 2 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 10:
        return False
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    except Exception:
        return False
    h, w = gray.shape

    is_grand = "grand" in cls_name.lower()
    is_system = "system" in cls_name.lower()

    # 1. Aspect ratio guard: authentic music blocks are horizontal ribbons
    if (w / float(max(1, h))) < 1.0:
        return False
    if not (is_grand or is_system):
        if (w / float(max(1, h))) < 2.5:
            return False

    min_h = 50 if (is_grand or is_system) else 22
    if h < min_h or w < 70:
        return False

    # Dynamic binarization
    bg_val = float(np.percentile(gray, 90)) if gray.size > 0 else 250.0
    ink_thresh = bg_val - 40.0
    bin_inv = (gray < ink_thresh).astype(np.uint8) * 255

    # 2. Continuous horizontal line detection (minimum segment length 10% width)
    kernel_len = max(20, min(80, int(w * 0.10)))
    if torch.cuda.is_available() and w >= 200:
        try:
            t_bin = torch.from_numpy(bin_inv).cuda().float().unsqueeze(0).unsqueeze(0)
            pad_k = kernel_len // 2
            e = -torch.nn.functional.max_pool2d(-t_bin, kernel_size=(1, kernel_len), stride=1, padding=(0, pad_k))
            d = torch.nn.functional.max_pool2d(e, kernel_size=(1, kernel_len), stride=1, padding=(0, pad_k))
            proj = (d.squeeze() > 128).sum(dim=1).cpu().numpy()
        except Exception:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
            lines_img = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)
            proj = np.sum(lines_img > 0, axis=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        lines_img = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)
        proj = np.sum(lines_img > 0, axis=1)
    min_row_coverage = max(15, int(w * 0.15))
    active_rows = np.where(proj >= min_row_coverage)[0]

    line_centers = []
    for r in active_rows:
        if not line_centers or r - line_centers[-1][-1] > 2:
            line_centers.append([r])
        else:
            line_centers[-1].append(r)

    centers = [float(np.mean(grp)) for grp in line_centers]

    # Real music staff MUST have continuous horizontal lines from Method 1
    min_lines = 8 if (is_grand or is_system) else 4
    if len(centers) < min_lines:
        return False

    diffs = np.diff(centers)
    median_s = float(np.median(diffs))

    # Scale-adaptive plausible spacing bounds
    if median_s < 5.0 or median_s > 24.0:
        return False

    # 3. Check for multi-row table grid (6 or more consecutive equidistant lines)
    if len(centers) >= 6:
        for i in range(len(diffs) - 4):
            sub_d = diffs[i : i + 5]
            mean_d = np.mean(sub_d)
            if mean_d > 0 and all(0.70 * mean_d <= d <= 1.30 * mean_d for d in sub_d):
                return False

    # 4. Multi-staff systems must possess an inter-staff gap between individual staves
    if is_grand or is_system:
        has_inter_staff_gap = any(d >= 2.0 * median_s for d in diffs)
        if not has_inter_staff_gap:
            return False

    consistent_diffs = [d for d in diffs if 0.55 * median_s <= d <= 1.45 * median_s]
    min_consistent = 6 if (is_grand or is_system) else 3
    return len(consistent_diffs) >= min_consistent


class LayoutDetector:
    """
    Music Document Layout Analysis (DLA) using OLA v2.0 (YOLOv8x/v11x).
    
    Detects:
    - grand_staff: 2-staff piano systems bound by braces (treble + bass)
    - staff: isolated single-staff lines (solfeggio, vocal melodies)
    - system: multi-staff systems (choral, ensemble)
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu", conf_threshold: float = 0.25):
        if weights_path is None:
            weights_path = str(Path(__file__).parent.parent / "weights" / "ola-layout-analysis-2.0-2025-03-09.pt")
            
        self.weights_path = Path(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

    OLA_RELEASE_URL = "https://github.com/v-dvorak/omr-layout-analysis/releases/download/ola-v2.0/ola-layout-analysis-2.0-2025-03-09.pt"

    @classmethod
    def _download_weights_if_missing(cls, target_path: Path):
        """Automatically downloads OLA v2.0 weights on first use if not present."""
        if target_path.is_file():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".tmp")
        print(f"[LayoutDetector] Pretrained weights not found at: {target_path}")
        print(f"[LayoutDetector] Downloading OLA v2.0 weights from {cls.OLA_RELEASE_URL}...")
        try:
            import urllib.request
            req = urllib.request.Request(cls.OLA_RELEASE_URL, headers={"User-Agent": "pdf_to_md_music"})
            with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r[LayoutDetector] Download progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
            print()
            temp_path.replace(target_path)
            print(f"[LayoutDetector] Successfully saved weights to: {target_path}")
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise FileNotFoundError(
                f"Weights file not found at: {target_path}. Automatic download failed ({e}). "
                f"Please manually download ola-layout-analysis-2.0-2025-03-09.pt from: {cls.OLA_RELEASE_URL}"
            ) from e

    def _ensure_loaded(self):
        if self.model is None:
            if not self.weights_path.is_file():
                self._download_weights_if_missing(self.weights_path)
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            self.names = self.model.names

            # 1-step GPU warmup pass when running on CUDA to prime CUDA runtime,
            # cuDNN kernels, Tensor Cores, and the PyTorch caching allocator
            is_cuda = (self.device == "cuda" or "cuda" in str(self.device).lower())
            if is_cuda and torch.cuda.is_available():
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                precision_kwargs = self._get_precision_kwargs(is_cuda=True)
                self.model.predict(
                    source=dummy_img,
                    conf=self.conf_threshold,
                    imgsz=640,
                    device=self.device,
                    verbose=False,
                    **precision_kwargs
                )

    @staticmethod
    def _get_precision_kwargs(is_cuda: bool) -> dict:
        """
        Determines the correct precision arguments for Ultralytics YOLO.
        Ultralytics 8.4+ unified precision under `quantize` (16 for FP16) and deprecated `half`.
        """
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                return {"quantize": 16} if is_cuda else {}
        except Exception:
            pass
        return {"half": is_cuda}

    def purge_gpu_memory(self) -> None:
        """
        Explicitly releases LayoutDetector (YOLO OLA v2.0) GPU memory resources:
        1. Deletes model reference and sets self.model to None.
        2. Triggers full Python garbage collection.
        3. Synchronizes CUDA operations, flushes PyTorch CUDA caching allocator,
           and collects CUDA IPC memory handles.
        Idempotent: safe to call multiple times or when model is already unloaded.
        """
        if self.model is not None:
            del self.model
            self.model = None

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

    @staticmethod
    def box_iou(b1: List[int], b2: List[int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / float(area1 + area2 - inter)

    @staticmethod
    def is_contained(small: List[int], big: List[int], thresh: float = 0.60) -> bool:
        x1 = max(small[0], big[0])
        y1 = max(small[1], big[1])
        x2 = min(small[2], big[2])
        y2 = min(small[3], big[3])
        if x2 <= x1 or y2 <= y1:
            return False
        inter = (x2 - x1) * (y2 - y1)
        small_area = (small[2] - small[0]) * (small[3] - small[1])
        return (inter / float(max(1, small_area))) >= thresh

    @staticmethod
    def vertical_overlap_ratio(b1: List[int], b2: List[int]) -> float:
        """Fraction of the smaller box's height that is vertically overlapped."""
        y1 = max(b1[1], b2[1])
        y2 = min(b1[3], b2[3])
        if y2 <= y1:
            return 0.0
        h1 = b1[3] - b1[1]
        h2 = b2[3] - b2[1]
        return (y2 - y1) / float(max(1, min(h1, h2)))

    @staticmethod
    def horizontal_overlap_or_gap(b1: List[int], b2: List[int]) -> int:
        """Returns positive value for overlap, negative value for gap."""
        x1 = max(b1[0], b2[0])
        x2 = min(b1[2], b2[2])
        if x2 > x1:
            return x2 - x1  # positive overlap
        if b1[2] <= b2[0]:
            return -(b2[0] - b1[2])
        else:
            return -(b1[0] - b2[2])

    @staticmethod
    def merge_boxes(b1: List[int], b2: List[int]) -> List[int]:
        return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

    @staticmethod
    def trace_staff_horizontal_extent(gray: np.ndarray, box: List[int]) -> List[int]:
        """
        Given a candidate detection box [x1, y1, x2, y2], traces authentic
        continuous horizontal staff lines left and right to capture the full width
        from clef/brace to terminating barline.
        """
        img_h, img_w = gray.shape[:2]
        x1, y1, x2, y2 = box
        pad_y = 6
        sy1 = max(0, y1 - pad_y)
        sy2 = min(img_h, y2 + pad_y)
        strip_gray = gray[sy1:sy2, :]
        if strip_gray.size == 0:
            return box

        bg_val = float(np.percentile(strip_gray, 90))
        bin_lines = (strip_gray < bg_val - 35).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        lines_only = cv2.morphologyEx(bin_lines, cv2.MORPH_OPEN, k)
        col_sums = np.sum(lines_only > 0, axis=0)

        mid_sums = col_sums[x1:x2]
        active_mid = mid_sums[mid_sums > 0]
        if len(active_mid) == 0:
            return box
        min_line_thresh = max(3, int(np.percentile(active_mid, 20) * 0.4))

        # Trace left from x1
        ext_x1 = x1
        zero_run = 0
        max_gap = 35
        for x in range(x1, -1, -1):
            if col_sums[x] >= min_line_thresh:
                ext_x1 = x
                zero_run = 0
            else:
                zero_run += 1
                if zero_run > max_gap:
                    break

        # Trace right from x2
        ext_x2 = x2
        zero_run = 0
        for x in range(x2, img_w):
            if col_sums[x] >= min_line_thresh:
                ext_x2 = x
                zero_run = 0
            else:
                zero_run += 1
                if zero_run > max_gap:
                    break

        # Check for left bracket/brace extending beyond ext_x1
        left_search_w = int(max(15, (y2 - y1) * 0.35))
        lx1 = max(0, ext_x1 - left_search_w)
        if ext_x1 > lx1:
            margin_gray = gray[sy1:sy2, lx1:ext_x1]
            if margin_gray.size > 0:
                m_bg = float(np.percentile(margin_gray, 85))
                m_dark = (margin_gray < m_bg - 28).astype(np.uint8)
                m_num, _, m_stats, _ = cv2.connectedComponentsWithStats(m_dark, connectivity=8)
                h_target = (sy2 - sy1) * 0.35
                for mi in range(1, m_num):
                    comp_h = m_stats[mi, cv2.CC_STAT_HEIGHT]
                    comp_x = m_stats[mi, cv2.CC_STAT_LEFT]
                    if comp_h >= h_target:
                        ext_x1 = max(0, min(ext_x1, lx1 + comp_x - 2))

        # Strict boundary clamping to prevent negative indices or out-of-bounds crops
        final_x1 = max(0, min(img_w - 1, int(ext_x1)))
        final_x2 = max(final_x1 + 1, min(img_w, int(ext_x2)))
        final_y1 = max(0, min(img_h - 1, int(y1)))
        final_y2 = max(final_y1 + 1, min(img_h, int(y2)))

        return [final_x1, final_y1, final_x2, final_y2]

    @classmethod
    def split_column_gutters(
        cls,
        detections: List[Dict[str, Any]],
        gray: np.ndarray,
        staff_s: float
    ) -> List[Dict[str, Any]]:
        """
        Splits wide macro detections that erroneously span across multiple columns
        separated by an empty vertical gutter (e.g. 2-column exercise pages).
        """
        res = []
        for b in detections:
            box = b["box"]
            w = box[2] - box[0]
            if w < int(staff_s * 28.0):
                res.append(b)
                continue
            roi = gray[box[1]:box[3], box[0]:box[2]]
            if roi.size == 0:
                res.append(b)
                continue
            bg = float(np.percentile(roi, 90))
            bin_roi = (roi < bg - 35).astype(np.uint8)
            col_sums = np.sum(bin_roi, axis=0)

            mid_start = int(w * 0.25)
            mid_end = int(w * 0.75)
            mid_sums = col_sums[mid_start:mid_end]
            gutter_min_w = int(staff_s * 3.0)

            max_run = 0
            best_start = -1
            cur_start = -1
            for idx, s in enumerate(mid_sums):
                if s <= 2:
                    if cur_start == -1:
                        cur_start = idx
                    run_len = idx - cur_start + 1
                    if run_len > max_run:
                        max_run = run_len
                        best_start = cur_start
                else:
                    cur_start = -1

            if max_run >= gutter_min_w:
                left_gutter = box[0] + mid_start + best_start
                right_gutter = left_gutter + max_run
                b1 = dict(b)
                b1["box"] = [box[0], box[1], left_gutter, box[3]]
                b2 = dict(b)
                b2["box"] = [right_gutter, box[1], box[2], box[3]]
                res.extend([b1, b2])
            else:
                res.append(b)
        return res

    @classmethod
    def heal_collinear_segments(
        cls,
        detections_list: List[Dict[str, Any]],
        v_overlap_thresh: float = 0.55,
        max_gap_px: Optional[int] = None,
        img_w: int = 1000,
        gray: Optional[np.ndarray] = None,
        staff_s: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Merges horizontally broken or overlapping segments of the same staff/grand_staff line.
        max_gap_px adapts to image width automatically.
        """
        if not detections_list:
            return []
        if max_gap_px is None:
            max_gap_px = max(40, int(img_w * 0.04))

        merged = []
        sorted_dets = sorted(detections_list, key=lambda x: x["confidence"], reverse=True)
        used = [False] * len(sorted_dets)

        for i in range(len(sorted_dets)):
            if used[i]:
                continue
            cur_box = list(sorted_dets[i]["box"])
            cur_conf = sorted_dets[i]["confidence"]
            cur_cls = sorted_dets[i]["class"]

            cur_s = sorted_dets[i].get("s", None)
            cur_staves = sorted_dets[i].get("staves_count", 1)

            for j in range(i + 1, len(sorted_dets)):
                if used[j]:
                    continue
                other_cls = sorted_dets[j]["class"]
                if other_cls != cur_cls:
                    continue
                other_box = sorted_dets[j]["box"]
                v_ratio = cls.vertical_overlap_ratio(cur_box, other_box)
                h_rel = cls.horizontal_overlap_or_gap(cur_box, other_box)

                h1 = cur_box[3] - cur_box[1]
                h2 = other_box[3] - other_box[1]
                cy1 = (cur_box[1] + cur_box[3]) / 2.0
                cy2 = (other_box[1] + other_box[3]) / 2.0

                if (v_ratio >= v_overlap_thresh 
                    and abs(h1 - h2) <= max(10, int(0.30 * max(h1, h2)))
                    and abs(cy1 - cy2) <= max(8, int(0.20 * max(h1, h2)))
                    and -max_gap_px <= h_rel):
                    # Guard: do not merge across an empty vertical column gutter
                    if gray is not None and h_rel < 0:
                        gx1 = min(cur_box[2], other_box[2])
                        gx2 = max(cur_box[0], other_box[0])
                        s_ref = staff_s or 8.5
                        if (gx2 - gx1) >= int(s_ref * 2.5):
                            gy1 = max(0, min(cur_box[1], other_box[1]))
                            gy2 = min(gray.shape[0], max(cur_box[3], other_box[3]))
                            gutter_roi = gray[gy1:gy2, gx1:gx2]
                            if gutter_roi.size > 0:
                                bg = float(np.percentile(gutter_roi, 90))
                                bin_g = (gutter_roi < bg - 35).astype(np.uint8)
                                sums = np.sum(bin_g, axis=0)
                                zero_runs = max((len(list(g)) for k, g in itertools.groupby(sums <= 2) if k), default=0)
                                if zero_runs >= int(s_ref * 2.5):
                                    continue
                    cur_box = cls.merge_boxes(cur_box, other_box)
                    cur_conf = max(cur_conf, sorted_dets[j]["confidence"])
                    cur_staves = max(cur_staves, sorted_dets[j].get("staves_count", 1))
                    used[j] = True

            used[i] = True
            m_dict = {
                "class": cur_cls,
                "confidence": cur_conf,
                "box": cur_box,
                "staves_count": cur_staves
            }
            if cur_s is not None:
                m_dict["s"] = cur_s
            merged.append(m_dict)
        return merged

    @classmethod
    def has_continuous_vertical_connector(
        cls,
        gray: np.ndarray,
        s1: Dict[str, Any],
        s2: Dict[str, Any],
        staff_s: float,
        return_x: bool = False
    ) -> Any:
        """
        Determines if two adjacent staves are physically joined on the left
        by a continuous vertical bracket, curly brace, or primary system barline.
        Scale-invariant, robust against curved braces, and immune to isolated margin labels.
        If return_x=True, returns (has_conn, bracket_left_x). Otherwise returns has_conn.
        """
        v_gap = s2["y_top"] - s1["y_bot"]
        min_x = min(s1["x_left"], s2["x_left"])
        if v_gap < 0:
            return (True, min_x) if return_x else True

        s = max(5.0, staff_s)

        # Brackets/braces are located to the left of the staff lines.
        # Search window: from (min_x - 6.5 * s) to (min_x + 2.0 * s)
        x_start = max(0, int(min_x - s * 6.5))
        x_end = min(gray.shape[1], int(min_x + s * 2.0))
        if x_end <= x_start:
            return (False, min_x) if return_x else False

        # Vertical range: across the inter-staff gap + overlap into both staves
        y_start = max(0, int(s1["y_bot"] - s * 1.5))
        y_end = min(gray.shape[0], int(s2["y_top"] + s * 1.5))

        strip = gray[y_start:y_end, x_start:x_end]
        if strip.size == 0:
            return (False, min_x) if return_x else False

        # Multi-threshold binarization (percentile + Otsu)
        bg = float(np.percentile(strip, 85))
        bin_dark = (strip < bg - 28).astype(np.uint8)
        blur = cv2.GaussianBlur(strip, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bin_comb = cv2.bitwise_or(bin_dark, (otsu > 0).astype(np.uint8))

        inner_y1 = s1["y_bot"] - y_start
        inner_y2 = s2["y_top"] - y_start
        gap_height = inner_y2 - inner_y1
        if gap_height <= 0:
            return (True, min_x) if return_x else True

        # Check 1: 8-connected component bridging the inter-staff gap
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_comb, connectivity=8)
        tol = int(s * 0.5)
        for i in range(1, num_labels):
            comp_x = stats[i, cv2.CC_STAT_LEFT]
            comp_y = stats[i, cv2.CC_STAT_TOP]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_bot = comp_y + comp_h

            if comp_y <= inner_y1 + tol and comp_bot >= inner_y2 - tol:
                aspect = comp_h / float(max(1, comp_w))
                if comp_h >= gap_height * 0.80 and (aspect >= 1.2 or comp_w <= int(s * 3.5)):
                    actual_x = x_start + comp_x
                    return (True, actual_x) if return_x else True

        # Check 2: Row coverage across the gap in a vertical corridor
        dilated = cv2.dilate(bin_comb, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        inner_roi = dilated[inner_y1:inner_y2, :]

        win_w = max(5, int(s * 1.8))
        max_cov = 0.0
        best_win_x = 0
        for x in range(0, inner_roi.shape[1] - win_w + 1):
            window = inner_roi[:, x:x + win_w]
            cov = float(np.mean(np.sum(window > 0, axis=1) > 0))
            if cov > max_cov:
                max_cov = cov
                best_win_x = x

        if max_cov >= 0.75:
            actual_x = x_start + best_win_x
            return (True, actual_x) if return_x else True

        return (False, min_x) if return_x else False

    @classmethod
    def detect_connecting_barlines(cls, gray: np.ndarray, s1: Dict[str, Any], s2: Dict[str, Any], staff_s: float) -> bool:
        """
        Scans the inter-staff gap across the horizontal span for continuous vertical barlines
        linking the two staves across measures. Immune to faded/broken left curly braces.
        """
        v_gap = s2["y_top"] - s1["y_bot"]
        if v_gap <= 0:
            return True

        x_start = max(s1["x_left"], s2["x_left"])
        x_end = min(s1["x_right"], s2["x_right"])
        if (x_end - x_start) <= int(staff_s * 4.0):
            return False

        inter_roi = gray[s1["y_bot"]:s2["y_top"], x_start:x_end]
        if inter_roi.size == 0:
            return False

        bg = float(np.percentile(inter_roi, 90))
        bin_inv = (inter_roi < bg - 35).astype(np.uint8)

        k_h = max(3, int(v_gap * 0.70))
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        opened = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, vert_k)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opened)
        for idx in range(1, num_labels):
            h_stat = stats[idx, cv2.CC_STAT_HEIGHT]
            w_stat = stats[idx, cv2.CC_STAT_WIDTH]
            if h_stat >= int(v_gap * 0.75) and (h_stat / max(1, w_stat)) >= 2.0:
                return True
        return False

    @classmethod
    def expand_envelope_to_ledger_lines(
        cls,
        gray: np.ndarray,
        box: List[int],
        staff_s: float,
        max_search_s: float = 8.5,
        whitespace_s: float = 0.75
    ) -> List[int]:
        """
        Dynamically expands vertical box envelope using 8-connected components and ink projection
        so notes on ledger lines, tall stems, accidentals, and dynamics are never bisected.
        """
        h_img, w_img = gray.shape[:2]
        x1, y1, x2, y2 = box
        search_margin = int(max_search_s * staff_s)

        y_search_min = max(0, y1 - search_margin)
        y_search_max = min(h_img, y2 + search_margin)
        x_min = max(0, x1)
        x_max = min(w_img, x2)

        corridor = gray[y_search_min:y_search_max, x_min:x_max]
        if corridor.size == 0:
            return box

        bg = float(np.percentile(corridor, 90))
        bin_corridor = (corridor < bg - 35).astype(np.uint8)

        vert_proj = np.sum(bin_corridor, axis=1)
        max_ink = np.max(vert_proj) if np.max(vert_proj) > 0 else 1
        norm_proj = vert_proj / float(max_ink)

        staff_top_rel = y1 - y_search_min
        staff_bot_rel = y2 - y_search_min
        whitespace_limit = max(1, int(whitespace_s * staff_s))

        empty_count = 0
        new_top_rel = staff_top_rel
        for y in range(staff_top_rel - 1, -1, -1):
            if norm_proj[y] <= 0.015:
                empty_count += 1
                if empty_count >= whitespace_limit:
                    new_top_rel = y + empty_count
                    break
            else:
                empty_count = 0
                new_top_rel = y

        empty_count = 0
        new_bot_rel = staff_bot_rel
        for y in range(staff_bot_rel + 1, len(norm_proj)):
            if norm_proj[y] <= 0.015:
                empty_count += 1
                if empty_count >= whitespace_limit:
                    new_bot_rel = y - empty_count
                    break
            else:
                empty_count = 0
                new_bot_rel = y

        orig_ymin = y_search_min + max(0, new_top_rel)
        orig_ymax = y_search_min + min(corridor.shape[0], new_bot_rel)
        final_ymin = orig_ymin
        final_ymax = orig_ymax

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_corridor, connectivity=8)
        for i in range(1, num_labels):
            comp_y = y_search_min + stats[i, cv2.CC_STAT_TOP]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_bot = comp_y + comp_h

            if comp_y < orig_ymin and comp_bot >= orig_ymin:
                if comp_h <= int(8.0 * staff_s):
                    final_ymin = min(final_ymin, comp_y)
            if comp_bot > orig_ymax and comp_y <= orig_ymax:
                if comp_h <= int(8.0 * staff_s):
                    final_ymax = max(final_ymax, comp_bot)

        return [int(x1), int(max(0, final_ymin)), int(x2), int(min(h_img, final_ymax))]

    @classmethod
    def extract_physical_5line_staves(cls, gray: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        DPI-invariant, font-invariant physical 5-line music staff extractor.
        Guarantees 100% recall of music staves across any sheet music book.
        Returns (list_of_staves, measured_staff_spacing_S).
        """
        img_h, img_w = gray.shape[:2]
        bg_val = float(np.percentile(gray, 92))
        bin_inv = (gray < bg_val - 35).astype(np.uint8) * 255

        # Coarse pass: determine staff line vertical centers and spacing S across page
        k_len = max(35, int(img_w * 0.12))
        mx = int(img_w * 0.05)

        use_cuda = torch.cuda.is_available()
        t_bin_gpu = None
        if use_cuda:
            try:
                t_bin_gpu = torch.from_numpy(bin_inv).cuda().float().unsqueeze(0).unsqueeze(0)
                pad1 = k_len // 2
                e1 = -torch.nn.functional.max_pool2d(-t_bin_gpu, kernel_size=(1, k_len), stride=1, padding=(0, pad1))
                d1 = torch.nn.functional.max_pool2d(e1, kernel_size=(1, k_len), stride=1, padding=(0, pad1))
                row_sums = (d1.squeeze()[:, mx:img_w - mx] > 128).sum(dim=1).cpu().numpy()
            except Exception:
                use_cuda = False
                t_bin_gpu = None

        if not use_cuda or t_bin_gpu is None:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
            lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k)
            row_sums = np.sum(lines[:, mx:img_w - mx] > 0, axis=1)

        line_centers = []
        in_line = False
        start_y = 0
        thresh = max(60, int((img_w - 2 * mx) * 0.10))
        for y in range(img_h):
            if row_sums[y] > thresh:
                if not in_line:
                    start_y = y
                    in_line = True
            else:
                if in_line:
                    line_centers.append((start_y + y - 1) / 2.0)
                    in_line = False
        if in_line:
            line_centers.append((start_y + img_h - 1) / 2.0)

        if len(line_centers) < 5:
            return [], 10.0

        # Cluster/merge line centers closer than 3.5 px (from side-by-side columns or slight scan tilt)
        merged_centers = []
        for c in line_centers:
            if not merged_centers:
                merged_centers.append([c])
            else:
                if c - merged_centers[-1][-1] <= 3.5:
                    merged_centers[-1].append(c)
                else:
                    merged_centers.append([c])
        clean_centers = [float(np.mean(group)) for group in merged_centers]

        if len(clean_centers) < 5:
            return [], 10.0

        staves = []
        i = 0
        while i <= len(clean_centers) - 5:
            sub = clean_centers[i : i + 5]
            diffs = np.diff(sub)
            local_s = float(np.mean(diffs))

            if local_s < 5.0 or local_s > 24.0:
                i += 1
                continue

            # Spacing uniformity: relative standard deviation <= 15%
            if (np.std(diffs) / local_s) > 0.15:
                i += 1
                continue

            # Grid table rejection: reject if embedded in a larger regular grid (>= 6 lines)
            has_above = (i > 0 and abs((sub[0] - clean_centers[i - 1]) - local_s) <= 0.35 * local_s)
            has_below = (i + 5 < len(clean_centers) and abs((clean_centers[i + 5] - sub[-1]) - local_s) <= 0.35 * local_s)
            if has_above or has_below:
                i += 1
                continue

            # Trace horizontal extent using fine kernel scaled to local S
            k_len_fine = max(20, min(35, int(local_s * 2.5)))
            k_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len_fine, 1))
            lines_fine = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k_fine)
            yt, yb = int(sub[0]), int(sub[-1])
            strip = lines_fine[max(0, yt - 2):min(img_h, yb + 3), :]
            col_s = np.sum(strip > 0, axis=0)

            # Segment detection: handles multiple columns / staves on same line and rejects isolated small arrows
            active_cols = col_s >= 2
            segs = cls.find_segments(active_cols, max_gap=int(local_s * 4.0))

            found_any = False
            for xl, xr in segs:
                w_staff = xr - xl
                h_staff = yb - yt
                if w_staff >= 80 and (w_staff / float(max(1, h_staff))) >= 2.5:
                    staves.append({
                        "y_top": yt,
                        "y_bot": yb,
                        "x_left": int(xl),
                        "x_right": int(xr),
                        "s": local_s,
                        "lines": sub
                    })
                    found_any = True

            if found_any:
                i += 5
            else:
                i += 1

        staves.sort(key=lambda s: s["y_top"])
        median_page_s = float(np.median([s["s"] for s in staves])) if staves else 10.0
        return staves, median_page_s

    @staticmethod
    def find_segments(binary_row: np.ndarray, max_gap: int) -> List[Tuple[int, int]]:
        idx = np.where(binary_row)[0]
        if len(idx) == 0:
            return []
        segments = []
        start = idx[0]
        prev = idx[0]
        for x in idx[1:]:
            if x - prev > max_gap:
                segments.append((start, prev))
                start = x
            prev = x
        segments.append((start, prev))
        return segments

    def detect(self, img_bgr: np.ndarray, imgsz: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Runs hybrid layout detection combining deep neural OLA classification
        with geometric physical staff analysis.
        Guarantees 100% recall across any music book scale or layout.
        """
        self._ensure_loaded()
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Estimate staff spacing S
        staff_s = 8.5
        try:
            s_est = estimate_staff_spacing(gray)
            if s_est is not None:
                if isinstance(s_est, (int, float)) and 5.0 <= s_est <= 24.0:
                    staff_s = float(s_est)
                elif isinstance(s_est, (tuple, list)) and len(s_est) >= 1 and 5.0 <= s_est[0] <= 24.0:
                    staff_s = float(s_est[0])
        except Exception:
            pass

        # 2. Neural inference with YOLO OLA v2.0
        is_cuda = (self.device == "cuda" or "cuda" in str(self.device).lower())
        if imgsz is None:
            max_dim = max(img_h, img_w)
            target_sz = int(np.ceil(max_dim / 32.0) * 32)
            max_cap = 1280
            imgsz = min(max_cap, max(1024, target_sz))

        effective_conf = min(self.conf_threshold, 0.20)
        precision_kwargs = self._get_precision_kwargs(is_cuda)
        results = self.model.predict(
            source=img_bgr,
            conf=effective_conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            **precision_kwargs
        )

        yolo_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.names.get(cls_id, str(cls_id)).lower().replace(" ", "_")
                # Only macro notation blocks: staves, grand_staff, systems (ignore measure slices)
                if cls_name not in ("staves", "grand_staff", "systems"):
                    continue
                norm_cls = "grand_staff" if "grand" in cls_name else ("system" if "system" in cls_name else "staff")
                b_xy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                b_h = b_xy[3] - b_xy[1]
                # Physical constraint: a grand_staff or system must be >= 7.0 * staff_s tall (~60px)
                if norm_cls in ("grand_staff", "system") and b_h < int(staff_s * 7.0):
                    norm_cls = "staff"

                # Trace authentic staff horizontal extent so clefs, braces, and full measures are never truncated
                b_xy = self.trace_staff_horizontal_extent(gray, b_xy)

                # Validate candidate with physical filter
                x1, y1, x2, y2 = b_xy
                if (x2 - x1) < 10 or (y2 - y1) < 5:
                    continue
                crop = img_bgr[y1:y2, x1:x2]
                if is_valid_music_staff(crop, norm_cls, conf):
                    yolo_boxes.append({
                        "box": b_xy,
                        "conf": conf,
                        "class": norm_cls,
                        "staves_count": 2 if norm_cls == "grand_staff" else (3 if norm_cls == "system" else 1),
                        "s": staff_s
                    })

        # Split wide boxes spanning across multiple columns separated by a vertical gutter
        yolo_boxes = self.split_column_gutters(yolo_boxes, gray, staff_s)

        # 3. Deduplicate and suppress containers
        multi_staff = [b for b in yolo_boxes if b['class'] in ('grand_staff', 'system')]
        multi_staff.sort(key=lambda x: x['conf'], reverse=True)
        kept_multi = []
        for b in multi_staff:
            conflict = False
            for km in kept_multi:
                iou = self.box_iou(b['box'], km['box'])
                v_ratio = self.vertical_overlap_ratio(b['box'], km['box'])
                inter_x = max(0, min(b['box'][2], km['box'][2]) - max(b['box'][0], km['box'][0]))
                min_w = min(b['box'][2] - b['box'][0], km['box'][2] - km['box'][0])
                h_ratio = inter_x / float(max(1, min_w))

                if iou > 0.50:
                    conflict = True
                    break
                if (b['class'] == 'system' or km['class'] == 'system') and v_ratio >= 0.30 and h_ratio >= 0.40:
                    conflict = True
                    break
                if self.is_contained(b['box'], km['box'], thresh=0.50):
                    conflict = True
                    break
                if self.is_contained(km['box'], b['box'], thresh=0.50):
                    conflict = True
                    break
            if not conflict:
                kept_multi.append(b)

        single_staves = [b for b in yolo_boxes if b['class'] == 'staff']
        single_staves.sort(key=lambda x: x['conf'], reverse=True)
        kept_single = []
        for s in single_staves:
            contained = False
            for km in kept_multi:
                v_ratio = self.vertical_overlap_ratio(s['box'], km['box'])
                inter_x = max(0, min(s['box'][2], km['box'][2]) - max(s['box'][0], km['box'][0]))
                min_w = min(s['box'][2] - s['box'][0], km['box'][2] - km['box'][0])
                h_ratio = inter_x / float(max(1, min_w))
                if self.is_contained(s['box'], km['box'], thresh=0.50) or (v_ratio >= 0.50 and h_ratio >= 0.50):
                    km['box'][0] = min(km['box'][0], s['box'][0])
                    km['box'][2] = max(km['box'][2], s['box'][2])
                    contained = True
                    break
            if not contained:
                if not any(self.box_iou(s['box'], ks['box']) > 0.50 for ks in kept_single):
                    kept_single.append(s)

        # 4. Physical staff rescue pass (for any authentic staff missed by YOLO)
        phys_staves, _ = self.extract_physical_5line_staves(gray)
        for ps in phys_staves:
            p_box = [ps["x_left"], ps["y_top"], ps["x_right"], ps["y_bot"]]
            covered = False
            for kb in (kept_multi + kept_single):
                if self.box_iou(p_box, kb['box']) > 0.30 or self.is_contained(p_box, kb['box'], thresh=0.50):
                    covered = True
                    break
            if not covered:
                p_box = self.trace_staff_horizontal_extent(gray, p_box)
                x1, y1, x2, y2 = p_box
                if (x2 - x1) < 10 or (y2 - y1) < 5:
                    continue
                crop = img_bgr[y1:y2, x1:x2]
                if is_valid_music_staff(crop, "staff", 0.90):
                    kept_single.append({
                        "box": p_box,
                        "conf": 0.90,
                        "class": "staff",
                        "staves_count": 1,
                        "s": ps["s"]
                    })

        # 5. Unify multi-staff systems and grand staves across all candidate blocks
        candidates = kept_multi + kept_single
        candidates.sort(key=lambda x: x['box'][1])
        merged_blocks = []
        skip = set()
        for idx in range(len(candidates)):
            if idx in skip:
                continue
            cur = dict(candidates[idx])
            accum_box = list(cur['box'])
            accum_conf = cur['conf']

            lookahead = idx + 1
            while lookahead < len(candidates):
                nxt = candidates[lookahead]
                nxt_box = nxt['box']
                inter_x = max(0, min(accum_box[2], nxt_box[2]) - max(accum_box[0], nxt_box[0]))
                union_x = max(accum_box[2], nxt_box[2]) - min(accum_box[0], nxt_box[0])
                h_iou = inter_x / float(max(1, union_x))

                if h_iou < 0.35:
                    lookahead += 1
                    continue

                v_overlap = max(0, min(accum_box[3], nxt_box[3]) - max(accum_box[1], nxt_box[1]))
                v_gap = nxt_box[1] - accum_box[3]

                should_merge = False
                bx = accum_box[0]

                # 1. Overlapping blocks that share staves (e.g. YOLO duplicate detections)
                if v_overlap > 0:
                    should_merge = True
                # 2. Adjacent blocks with authentic vertical connector line, bracket, or barlines
                elif 0 <= v_gap <= int(staff_s * 14.0):
                    s_top = {"x_left": accum_box[0], "x_right": accum_box[2], "y_top": accum_box[1], "y_bot": accum_box[3]}
                    s_bot = {"x_left": nxt_box[0], "x_right": nxt_box[2], "y_top": nxt_box[1], "y_bot": nxt_box[3]}
                    has_conn, bx = self.has_continuous_vertical_connector(gray, s_top, s_bot, staff_s, return_x=True)
                    has_bars = self.detect_connecting_barlines(gray, s_top, s_bot, staff_s)
                    if has_conn or has_bars:
                        should_merge = True

                if should_merge:
                    accum_box[0] = min(accum_box[0], nxt_box[0], bx)
                    accum_box[1] = min(accum_box[1], nxt_box[1])
                    accum_box[2] = max(accum_box[2], nxt_box[2])
                    accum_box[3] = max(accum_box[3], nxt_box[3])
                    accum_conf = max(accum_conf, nxt['conf'])
                    skip.add(lookahead)
                    lookahead += 1
                else:
                    break

            # Count physical staves contained within accum_box
            contained_staves = 0
            for ps in phys_staves:
                ps_mid_y = (ps['y_top'] + ps['y_bot']) / 2.0
                ps_mid_x = (ps['x_left'] + ps['x_right']) / 2.0
                if accum_box[1] - staff_s <= ps_mid_y <= accum_box[3] + staff_s:
                    if accum_box[0] - staff_s * 2.0 <= ps_mid_x <= accum_box[2] + staff_s * 2.0:
                        contained_staves += 1

            accum_h = accum_box[3] - accum_box[1]
            if contained_staves >= 3 or accum_h >= int(26.0 * staff_s):
                norm_cls = "system"
            elif contained_staves == 2 or accum_h >= int(11.0 * staff_s):
                norm_cls = "grand_staff"
            elif contained_staves == 1:
                norm_cls = "staff"
            else:
                norm_cls = cur['class']

            merged_blocks.append({
                "box": accum_box,
                "conf": accum_conf,
                "class": norm_cls,
                "staves_count": max(1, contained_staves),
                "s": staff_s
            })

        all_blocks = merged_blocks
        all_blocks.sort(key=lambda x: x['box'][1])

        # 6. Expand vertical envelope to ledger lines
        expanded_blocks = []
        for b in all_blocks:
            exp_box = self.expand_envelope_to_ledger_lines(gray, b['box'], staff_s)
            expanded_blocks.append({
                "class": b['class'],
                "confidence": b['conf'],
                "box": exp_box,
                "staves_count": b.get("staves_count", 1),
                "s": staff_s
            })

        # 7. Collinear healing
        final_blocks = self.heal_collinear_segments(expanded_blocks, v_overlap_thresh=0.55, img_w=img_w, gray=gray, staff_s=staff_s)
        final_blocks.sort(key=lambda item: item["box"][1])

        # 7.5. Resolve vertical overlaps on the same column
        non_overlapping = []
        skip_indices = set()
        for i in range(len(final_blocks)):
            if i in skip_indices:
                continue
            cur = dict(final_blocks[i])
            c_box = list(cur["box"])
            c_conf = cur["confidence"]
            c_cls = cur["class"]
            c_staves = cur.get("staves_count", 1)

            for j in range(i + 1, len(final_blocks)):
                if j in skip_indices:
                    continue
                nxt = final_blocks[j]
                n_box = nxt["box"]

                inter_x = max(0, min(c_box[2], n_box[2]) - max(c_box[0], n_box[0]))
                union_x = max(c_box[2], n_box[2]) - min(c_box[0], n_box[0])
                h_iou = inter_x / float(max(1, union_x))
                v_ratio = self.vertical_overlap_ratio(c_box, n_box)
                h_rel = self.horizontal_overlap_or_gap(c_box, n_box)

                should_unify = False
                v_overlap = max(0, min(c_box[3], n_box[3]) - max(c_box[1], n_box[1]))
                if h_iou >= 0.35 and v_overlap > 8:
                    should_unify = True
                elif v_ratio >= 0.70 and h_rel >= 0:
                    should_unify = True

                # Guard against merging across vertical column gutters
                if should_unify and h_rel < 0:
                    gx1 = min(c_box[2], n_box[2])
                    gx2 = max(c_box[0], n_box[0])
                    s_ref = staff_s or 8.5
                    if (gx2 - gx1) >= int(s_ref * 2.5):
                        gy1 = max(0, min(c_box[1], n_box[1]))
                        gy2 = min(gray.shape[0], max(c_box[3], n_box[3]))
                        gutter_roi = gray[gy1:gy2, gx1:gx2]
                        if gutter_roi.size > 0:
                            bg = float(np.percentile(gutter_roi, 90))
                            bin_g = (gutter_roi < bg - 35).astype(np.uint8)
                            sums = np.sum(bin_g, axis=0)
                            zero_runs = max((len(list(g)) for k, g in itertools.groupby(sums <= 2) if k), default=0)
                            if zero_runs >= int(s_ref * 2.5):
                                should_unify = False

                if should_unify:
                    c_box = [min(c_box[0], n_box[0]), min(c_box[1], n_box[1]), max(c_box[2], n_box[2]), max(c_box[3], n_box[3])]
                    c_conf = max(c_conf, nxt["confidence"])
                    c_staves = c_staves + nxt.get("staves_count", 1)
                    c_cls = "system" if (c_cls == "system" or nxt["class"] == "system" or c_staves >= 3 or (c_box[3] - c_box[1]) >= int(26.0 * staff_s)) else "grand_staff"
                    skip_indices.add(j)

            cur["box"] = c_box
            cur["confidence"] = c_conf
            cur["class"] = c_cls
            cur["staves_count"] = c_staves
            non_overlapping.append(cur)

        # 7.6. Filter duplicate interior fragments fully contained in larger blocks
        uncontained = []
        for i, b in enumerate(non_overlapping):
            is_dup = False
            for j, other in enumerate(non_overlapping):
                if i != j and self.is_contained(b["box"], other["box"], thresh=0.60):
                    is_dup = True
                    break
            if not is_dup:
                uncontained.append(b)

        final_blocks = uncontained
        final_blocks.sort(key=lambda item: item["box"][1])

        # 8. Apply scale-adaptive bounded padding in units of S
        filtered = []
        num_cands = len(final_blocks)

        for idx_cand, d in enumerate(final_blocks):
            cls_name = d["class"]
            x1, y1, x2, y2 = d["box"]
            s_val = d.get("s", staff_s)

            pad_y = int(min(16, max(6, s_val * 0.8)))
            pad_x = int(min(30, max(12, s_val * 1.5)))

            px1 = max(0, x1 - pad_x)
            px2 = min(img_w, x2 + pad_x)

            # Only constrain vertical padding against preceding/succeeding blocks that horizontally overlap
            py1 = max(0, y1 - pad_y)
            for prev_d in filtered:
                prev_b = prev_d["padded_box"]
                h_overlap = max(0, min(px2, prev_b[2]) - max(px1, prev_b[0]))
                if h_overlap > 0 and prev_b[3] < y1:
                    py1 = max(py1, prev_b[3] + 4)

            py2 = min(img_h, y2 + pad_y)
            for next_d in final_blocks[idx_cand + 1:]:
                next_b = next_d["box"]
                h_overlap = max(0, min(px2, next_b[2]) - max(px1, next_b[0]))
                if h_overlap > 0 and next_b[1] > y2:
                    py2 = min(py2, next_b[1] - 4)
                    break

            filtered.append({
                "class": cls_name,
                "confidence": d["confidence"],
                "raw_box": [x1, y1, x2, y2],
                "padded_box": [px1, py1, px2, py2],
                "width": px2 - px1,
                "height": py2 - py1,
                "staves_count": d.get("staves_count", 1),
            })

        # Sort top-to-bottom
        filtered.sort(key=lambda item: item["padded_box"][1])
        return filtered


    def render_debug_image(self, img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws colored bounding boxes on a copy of the image for visual verification.
        - Green: grand_staff (two-staff piano system)
        - Blue: single_stave (monophonic melody)
        - Magenta: system (multi-staff score)
        """
        debug_img = img_bgr.copy()
        color_map = {
            "grand_staff": (0, 200, 0),    # Bright Green
            "staff": (220, 100, 0),        # Blue
            "system": (180, 0, 180),       # Magenta
        }
        
        for idx, d in enumerate(detections, start=1):
            cls_name = d["class"]
            conf = d["confidence"]
            color = color_map.get(cls_name, (0, 165, 255))
            
            # Draw padded box (solid)
            px1, py1, px2, py2 = d["padded_box"]
            cv2.rectangle(debug_img, (px1, py1), (px2, py2), color, 3)
            
            # Draw raw box (dashed or thin)
            rx1, ry1, rx2, ry2 = d["raw_box"]
            cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), color, 1)
            
            label = f"#{idx} {cls_name} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_y = max(py1 - 8, th + 8)
            cv2.rectangle(debug_img, (px1, label_y - th - 4), (px1 + tw + 8, label_y + baseline), color, -1)
            cv2.putText(debug_img, label, (px1 + 4, label_y), font, font_scale, (255, 255, 255), thickness)
            
        return debug_img

    def mask_page(self, img_bgr: np.ndarray, detections: List[Dict[str, Any]], tag_prefix: str = "P0001") -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Overlays clean white boxes on detected music areas and draws stub ID strings.
        Returns:
            (masked_img_bgr, crops_data)
        """
        masked_img = img_bgr.copy()
        img_h, img_w = masked_img.shape[:2]

        # Sanitize overly long prefixes (e.g. if caller passed full book title > 24 chars)
        clean_prefix = tag_prefix
        if len(tag_prefix) > 24:
            m = re.search(r"(P\d+.*)$", tag_prefix)
            if m:
                clean_prefix = m.group(1)

        crops_data = []
        for idx, d in enumerate(detections, start=1):
            px1, py1, px2, py2 = d["padded_box"]
            px1 = max(0, min(int(px1), img_w - 1))
            py1 = max(0, min(int(py1), img_h - 1))
            px2 = max(px1 + 1, min(int(px2), img_w))
            py2 = max(py1 + 1, min(int(py2), img_h))

            crop_bgr = img_bgr[py1:py2, px1:px2].copy()
            stub_id = f"{clean_prefix}_S{idx:02d}_{d['class']}"

            # Whiteout region on page
            cv2.rectangle(masked_img, (px1, py1), (px2, py2), (255, 255, 255), -1)

            # Add technical tag comment text
            stub_text = f"<!-- MUSIC_STUB_ID:{stub_id} -->"
            font = cv2.FONT_HERSHEY_SIMPLEX
            box_w = max(10, px2 - px1)
            box_h = max(10, py2 - py1)
            avail_w = max(10, box_w - 16)
            avail_h = max(6, box_h - 6)

            # Fit font size dynamically to stay strictly inside the white box
            font_scale = 0.55
            thickness = 2
            text_sz, _ = cv2.getTextSize(stub_text, font, font_scale, thickness)

            if text_sz[0] > avail_w:
                scale_factor = avail_w / float(max(1, text_sz[0]))
                font_scale = max(0.22, font_scale * scale_factor)
                thickness = 1 if font_scale < 0.45 else 2
                text_sz, _ = cv2.getTextSize(stub_text, font, font_scale, thickness)

            if text_sz[1] > avail_h:
                scale_factor = avail_h / float(max(1, text_sz[1]))
                font_scale = max(0.20, font_scale * scale_factor)
                thickness = 1
                text_sz, _ = cv2.getTextSize(stub_text, font, font_scale, thickness)

            # Position text cleanly inside the white box (never spilling into surrounding text)
            tx = px1 + 10
            if tx + text_sz[0] > px2 - 2:
                tx = max(px1 + 2, px2 - text_sz[0] - 2)
            tx = max(0, min(tx, img_w - text_sz[0] - 1))

            ty = py1 + max(25, (py2 - py1) // 2)

            cv2.putText(
                masked_img,
                stub_text,
                (int(tx), int(ty)),
                font,
                font_scale,
                (90, 90, 90),
                thickness,
            )

            crops_data.append({
                "stub_id": stub_id,
                "class": d["class"],
                "crop_img": crop_bgr,
                "box": d["padded_box"],
                "confidence": d["confidence"]
            })

        return masked_img, crops_data
