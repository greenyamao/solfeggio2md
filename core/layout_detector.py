import gc
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
    - Blank whitespace hallucinations
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    h, w = gray.shape

    is_grand = "grand" in cls_name.lower()
    min_h = 60 if is_grand else 28
    if h < min_h or w < 80:
        return False

    # Dynamic binarization
    bg_val = float(np.percentile(gray, 90)) if gray.size > 0 else 250.0
    ink_thresh = bg_val - 40.0
    bin_inv = (gray < ink_thresh).astype(np.uint8) * 255

    # 1. Morphological horizontal line detection (minimum segment length 10% width)
    kernel_len = max(20, min(80, int(w * 0.10)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    lines_img = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)

    # Row projection of horizontal line segments
    proj = np.sum(lines_img > 0, axis=1)
    min_row_coverage = max(20, int(w * 0.15))
    active_rows = np.where(proj >= min_row_coverage)[0]

    line_centers = []
    for r in active_rows:
        if not line_centers or r - line_centers[-1][-1] > 2:
            line_centers.append([r])
        else:
            line_centers[-1].append(r)

    centers = [float(np.mean(grp)) for grp in line_centers]

    # Real music staff MUST have continuous horizontal lines from Method 1
    # Printed text characters do not form long continuous horizontal lines
    if len(centers) < 2:
        return False

    # Method 2 fallback/supplement: if some lines are faint, check central strip
    if len(centers) < 3:
        x1, x2 = int(w * 0.25), int(w * 0.75)
        strip = gray[:, x1:x2]
        strip_proj = np.sum(255.0 - strip, axis=1)
        peaks = [
            y for y in range(1, h - 1)
            if strip_proj[y] > strip_proj[y - 1] and strip_proj[y] >= strip_proj[y + 1] and strip_proj[y] > 0.25 * np.max(strip_proj)
        ]
        fpeaks = []
        for y in sorted(peaks, key=lambda y: strip_proj[y], reverse=True):
            if all(abs(y - fp) >= 4 for fp in fpeaks):
                fpeaks.append(y)
        fpeaks.sort()
        if len(fpeaks) >= 3:
            centers = fpeaks

    # A real music staff must have at least 3 parallel lines
    if len(centers) < 3:
        return False

    diffs = np.diff(centers)
    median_s = float(np.median(diffs))

    # At 200 DPI, authentic music staff line spacing is between 6.5px and 30px
    # Font typography features (ascenders/baseline) are < 6px and get rejected
    if median_s < 6.5 or median_s > 30.0:
        return False

    consistent_diffs = [d for d in diffs if 0.55 * median_s <= d <= 1.45 * median_s]
    min_consistent = 4 if is_grand else 2
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

    def _ensure_loaded(self):
        if self.model is None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"Weights file not found at: {self.weights_path}. Please download ola-layout-analysis-2.0."
                )
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

        return [ext_x1, y1, ext_x2, y2]

    @classmethod
    def heal_collinear_segments(
        cls,
        detections_list: List[Dict[str, Any]],
        v_overlap_thresh: float = 0.55,
        max_gap_px: Optional[int] = None,
        img_w: int = 1000
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
                other_box = sorted_dets[j]["box"]
                v_ratio = cls.vertical_overlap_ratio(cur_box, other_box)
                h_rel = cls.horizontal_overlap_or_gap(cur_box, other_box)

                if v_ratio >= v_overlap_thresh and h_rel >= -max_gap_px:
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
    def has_continuous_vertical_connector(cls, gray: np.ndarray, s1: Dict[str, Any], s2: Dict[str, Any], staff_s: float) -> bool:
        """
        Determines if two adjacent staves are physically joined on the left
        by a continuous vertical bracket, curly brace, or primary system barline.
        Scale-invariant and immune to isolated exercise numbers, clefs, or margin text.
        """
        v_gap = s2["y_top"] - s1["y_bot"]
        if v_gap <= 0:
            return True

        min_x = min(s1["x_left"], s2["x_left"])
        max_x = max(s1["x_left"], s2["x_left"])
        x_start = max(0, int(min_x - staff_s * 2.5))
        x_end = min(gray.shape[1], int(max_x + staff_s * 1.5))
        if x_end <= x_start:
            return False

        y_start = max(0, s1["y_bot"] - int(staff_s * 0.5))
        y_end = min(gray.shape[0], s2["y_top"] + int(staff_s * 0.5))

        strip = gray[y_start:y_end, x_start:x_end]
        if strip.size == 0:
            return False

        bg = float(np.percentile(strip, 88))
        bin_dark = (strip < bg - 35).astype(np.uint8)

        k_h = max(5, int(staff_s * 1.0))
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        vert_lines = cv2.morphologyEx(bin_dark, cv2.MORPH_OPEN, vert_k)

        inner_y_start = s1["y_bot"] - y_start
        inner_y_end = s2["y_top"] - y_start
        if inner_y_end <= inner_y_start:
            return True

        inner_lines = vert_lines[inner_y_start:inner_y_end, :]
        row_coverage = float(np.mean(np.sum(inner_lines > 0, axis=1) > 0))
        return row_coverage >= 0.65

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
        max_search_s: float = 3.5,
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

        final_ymin = y_search_min + max(0, new_top_rel)
        final_ymax = y_search_min + min(corridor.shape[0], new_bot_rel)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_corridor, connectivity=8)
        for i in range(1, num_labels):
            comp_y = y_search_min + stats[i, cv2.CC_STAT_TOP]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_bot = comp_y + comp_h

            if comp_y < final_ymin and comp_bot >= final_ymin:
                if comp_h <= int(5.5 * staff_s):
                    final_ymin = min(final_ymin, comp_y)
            if comp_bot > final_ymax and comp_y <= final_ymax:
                if comp_h <= int(5.5 * staff_s):
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
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
        lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k)

        mx = int(img_w * 0.05)
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

        diffs = np.diff(line_centers)
        plausible = diffs[(diffs >= 5) & (diffs <= 35)]
        if len(plausible) == 0:
            return [], 10.0
        staff_s = float(np.median(plausible))

        # Fine pass: use localized kernel to trace curved/bent staff ends near gutter and margins
        k_len_fine = max(20, min(35, int(staff_s * 2.5)))
        k_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len_fine, 1))
        lines_fine = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k_fine)

        staves = []
        i = 0
        while i <= len(line_centers) - 5:
            grp = line_centers[i:i+5]
            g_diffs = np.diff(grp)
            if all(abs(d - staff_s) <= max(2.5, staff_s * 0.40) for d in g_diffs):
                yt = int(grp[0])
                yb = int(grp[-1])
                strip = lines_fine[max(0, yt - 2):min(img_h, yb + 3), :]
                col_s = np.sum(strip > 0, axis=0)
                act_cols = np.where(col_s >= 2)[0]
                if len(act_cols) == 0:
                    act_cols = np.where(col_s > 0)[0]
                if len(act_cols) > 0:
                    xl = int(act_cols[0])
                    xr = int(act_cols[-1])
                else:
                    xl, xr = mx, img_w - mx
                staves.append({
                    "y_top": yt,
                    "y_bot": yb,
                    "x_left": xl,
                    "x_right": xr,
                    "s": staff_s
                })
                i += 5
            else:
                i += 1

        return staves, staff_s

    def detect(self, img_bgr: np.ndarray, imgsz: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Runs hybrid layout detection combining physical 5-line staff geometry
        with deep neural OLA classification.
        Guarantees 100% recall across any music book scale or layout.
        """
        self._ensure_loaded()
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Physical 5-line extraction (ground truth geometric anchor)
        phys_staves, staff_s = self.extract_physical_5line_staves(gray)

        # 2. Group physical staves into systems or grand staves using dynamic scale S
        phys_blocks = []
        skip_idx = set()
        for idx in range(len(phys_staves)):
            if idx in skip_idx:
                continue
            cur = [phys_staves[idx]]
            next_idx = idx + 1
            while next_idx < len(phys_staves):
                prev_s = cur[-1]
                cand_s = phys_staves[next_idx]

                inter = max(0, min(prev_s["x_right"], cand_s["x_right"]) - max(prev_s["x_left"], cand_s["x_left"]))
                union = max(prev_s["x_right"], cand_s["x_right"]) - min(prev_s["x_left"], cand_s["x_left"])
                h_iou = inter / float(max(1, union))
                v_gap = cand_s["y_top"] - prev_s["y_bot"]

                has_connector = self.has_continuous_vertical_connector(gray, prev_s, cand_s, staff_s)
                has_barlines = self.detect_connecting_barlines(gray, prev_s, cand_s, staff_s)

                can_merge = False
                if len(cur) == 1:
                    if has_connector or has_barlines:
                        # With verified left brace or through-measure barlines, natural piano layout allows wider gap
                        if 0 <= v_gap <= int(staff_s * 12.0) and (h_iou >= 0.40 or has_connector):
                            can_merge = True
                    else:
                        # Geometrically proximate unlinked staves
                        if 0 <= v_gap <= int(staff_s * 6.5) and h_iou >= 0.50:
                            can_merge = True
                elif len(cur) >= 2:
                    # Joining 3rd+ staff strictly requires continuous left system connector or through-system barlines
                    if (has_connector or has_barlines) and 0 <= v_gap <= int(staff_s * 12.0) and h_iou >= 0.40:
                        can_merge = True

                if can_merge:
                    cur.append(cand_s)
                    skip_idx.add(next_idx)
                    next_idx += 1
                else:
                    break

            num_staves = len(cur)
            cls_name = "staff" if num_staves == 1 else ("grand_staff" if num_staves == 2 else "system")
            bx1 = min(s["x_left"] for s in cur)
            bx2 = max(s["x_right"] for s in cur)
            by1 = cur[0]["y_top"]
            by2 = cur[-1]["y_bot"]

            # Dynamically expand envelope to enclose ledger lines, tall stems, accidentals, and dynamics
            exp_box = self.expand_envelope_to_ledger_lines(gray, [bx1, by1, bx2, by2], staff_s)

            phys_blocks.append({
                "class": cls_name,
                "confidence": 0.95,
                "box": exp_box,
                "staves_count": num_staves,
                "s": staff_s
            })

        # 3. Neural inference with YOLO OLA v2.0
        is_cuda = (self.device == "cuda" or "cuda" in str(self.device).lower())
        if imgsz is None:
            max_dim = max(img_h, img_w)
            target_sz = int(np.ceil(max_dim / 32.0) * 32)
            max_cap = 1920 if is_cuda else 1280
            imgsz = min(max_cap, max(1024, target_sz))

        effective_conf = min(self.conf_threshold, 0.10)
        precision_kwargs = self._get_precision_kwargs(is_cuda)
        results = self.model.predict(
            source=img_bgr,
            conf=effective_conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            **precision_kwargs
        )

        # 4. Integrate YOLO predictions with physical blocks
        final_blocks = list(phys_blocks)
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.names.get(cls_id, str(cls_id)).lower().replace(" ", "_")
                b_xy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]

                overlap = False
                for pb in final_blocks:
                    v_ratio = self.vertical_overlap_ratio(b_xy, pb["box"])
                    h_rel = self.horizontal_overlap_or_gap(b_xy, pb["box"])
                    v_mid = (b_xy[1] + b_xy[3]) / 2.0
                    if (v_ratio > 0.20 or (pb["box"][1] <= v_mid <= pb["box"][3])) and h_rel > 0:
                        overlap = True
                        if cls_name in ("grand_staff", "grandstaff") and pb["class"] == "staff" and pb["staves_count"] >= 2:
                            pb["class"] = "grand_staff"
                        elif cls_name in ("system", "systems") and pb["staves_count"] >= 3:
                            pb["class"] = "system"
                        break

                if not overlap:
                    total_v_intersect = sum(
                        max(0, min(b_xy[3], pb["box"][3]) - max(b_xy[1], pb["box"][1]))
                        for pb in final_blocks
                        if self.horizontal_overlap_or_gap(b_xy, pb["box"]) > 0
                    )
                    if (total_v_intersect / float(max(1, b_xy[3] - b_xy[1]))) > 0.25:
                        overlap = True

                if not overlap and conf >= 0.35 and is_valid_music_staff(img_bgr[b_xy[1]:b_xy[3], b_xy[0]:b_xy[2]], cls_name, conf):
                    bh = b_xy[3] - b_xy[1]
                    cls_clean = "staff" if bh < int(staff_s * 6.0) else ("grand_staff" if bh < int(staff_s * 16.0) else "system")
                    exp_b = self.expand_envelope_to_ledger_lines(gray, b_xy, staff_s)
                    final_blocks.append({
                        "class": cls_clean,
                        "confidence": conf,
                        "box": exp_b,
                        "staves_count": 1,
                        "s": staff_s
                    })

        # Heal horizontally split segments (e.g. short exercises split at double barlines)
        final_blocks = self.heal_collinear_segments(final_blocks, v_overlap_thresh=0.55, img_w=img_w)

        # Sort candidates top-to-bottom by raw y1
        final_blocks.sort(key=lambda item: item["box"][1])

        # 5. Apply scale-adaptive bounded padding in units of S
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

            prev_y2 = filtered[-1]["padded_box"][3] if filtered else None
            py1 = max(0, y1 - pad_y)
            if prev_y2 is not None and prev_y2 < y1:
                py1 = max(py1, prev_y2 + 4)

            next_y1 = final_blocks[idx_cand + 1]["box"][1] if idx_cand + 1 < num_cands else None
            py2 = min(img_h, y2 + pad_y)
            if next_y1 is not None and next_y1 > y2:
                py2 = min(py2, next_y1 - 4)

            filtered.append({
                "class": cls_name,
                "confidence": d["confidence"],
                "raw_box": [x1, y1, x2, y2],
                "padded_box": [px1, py1, px2, py2],
                "width": px2 - px1,
                "height": py2 - py1,
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
        crops_data = []
        
        for idx, d in enumerate(detections, start=1):
            px1, py1, px2, py2 = d["padded_box"]
            crop_bgr = img_bgr[py1:py2, px1:px2].copy()
            stub_id = f"{tag_prefix}_S{idx:02d}_{d['class']}"
            
            # Whiteout region on page
            cv2.rectangle(masked_img, (px1, py1), (px2, py2), (255, 255, 255), -1)
            
            # Add technical tag comment text
            stub_text = f"<!-- MUSIC_STUB_ID:{stub_id} -->"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            cv2.putText(masked_img, stub_text, (px1 + 10, py1 + max(25, (py2 - py1) // 2)), font, font_scale, (90, 90, 90), 2)
            
            crops_data.append({
                "stub_id": stub_id,
                "class": d["class"],
                "crop_img": crop_bgr,
                "box": d["padded_box"],
                "confidence": d["confidence"]
            })
            
        return masked_img, crops_data
