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

            for j in range(i + 1, len(sorted_dets)):
                if used[j]:
                    continue
                other_box = sorted_dets[j]["box"]
                v_ratio = cls.vertical_overlap_ratio(cur_box, other_box)
                h_rel = cls.horizontal_overlap_or_gap(cur_box, other_box)

                if v_ratio >= v_overlap_thresh and h_rel >= -max_gap_px:
                    cur_box = cls.merge_boxes(cur_box, other_box)
                    cur_conf = max(cur_conf, sorted_dets[j]["confidence"])
                    used[j] = True

            used[i] = True
            merged.append({
                "class": cur_cls,
                "confidence": cur_conf,
                "box": cur_box
            })
        return merged

    def detect(self, img_bgr: np.ndarray, imgsz: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Runs layout detection on an image with automatic scale adaptation.
        Returns a list of structured detection dictionaries.
        """
        self._ensure_loaded()
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Scale-adaptive YOLO inference size:
        # Preserves fine 1-px staff lines across different page resolutions (150 - 400 DPI)
        is_cuda = (self.device == "cuda" or "cuda" in str(self.device).lower())
        if imgsz is None:
            max_dim = max(img_h, img_w)
            target_sz = int(np.ceil(max_dim / 32.0) * 32)
            max_cap = 1920 if is_cuda else 1280
            imgsz = min(max_cap, max(1024, target_sz))

        # Inference with Ultralytics YOLO (FP16 enabled on CUDA, disabled on CPU)
        # Use conf=0.10 to capture fainter measures and staves, filtered by is_valid_music_staff
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

        raw_detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.names.get(cls_id, str(cls_id))

                raw_detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],  # x1, y1, x2, y2
                })

        # Process all 5 YOLO classes:
        # Extend each detection along authentic continuous horizontal staff lines
        # Classify candidates strictly by physical staff height
        raw_grand = []
        raw_staves = []
        raw_systems = []

        for d in raw_detections:
            cls_name = d["class"].lower().replace(" ", "_")
            traced_box = self.trace_staff_horizontal_extent(gray, d["box"])
            bh = traced_box[3] - traced_box[1]

            # Discard spurious measure boxes that bridge across multiple systems
            if "measure" in cls_name and bh > 170 and d["confidence"] < 0.85:
                continue

            d_traced = {
                "class": cls_name,
                "confidence": d["confidence"],
                "box": traced_box
            }

            if cls_name in ("grand_staff", "grandstaff"):
                if bh > 175:
                    d_traced["class"] = "system"
                    raw_systems.append(d_traced)
                else:
                    d_traced["class"] = "grand_staff"
                    raw_grand.append(d_traced)
            elif cls_name in ("systems", "system", "system_measures"):
                if bh >= 175:
                    d_traced["class"] = "system"
                    raw_systems.append(d_traced)
                elif bh >= 70:
                    d_traced["class"] = "grand_staff"
                    raw_grand.append(d_traced)
                else:
                    d_traced["class"] = "staff"
                    raw_staves.append(d_traced)
            elif cls_name in ("staves", "staff", "stave_measures"):
                d_traced["class"] = "staff"
                raw_staves.append(d_traced)

        box_iou = self.box_iou
        vertical_overlap_ratio = self.vertical_overlap_ratio
        horizontal_overlap_or_gap = self.horizontal_overlap_or_gap
        merge_boxes = self.merge_boxes
        heal_collinear_segments = lambda dets, **kw: self.heal_collinear_segments(dets, img_w=img_w, **kw)

        # 1. Process grand_staff (piano 2-staff systems)
        accepted_grand = heal_collinear_segments(raw_grand)
        for g in accepted_grand:
            g["class"] = "grand_staff"

        # 2. Process systems (multi-staff orchestral/chamber scores)
        accepted_systems = heal_collinear_segments(raw_systems)
        for sys_b in accepted_systems:
            sys_b["class"] = "system"

        # 3. Process staves: remove any staff that falls inside a grand_staff or system
        staves_outside = []
        for s in raw_staves:
            inside = False
            for g in accepted_grand:
                if vertical_overlap_ratio(s["box"], g["box"]) > 0.60 and horizontal_overlap_or_gap(s["box"], g["box"]) > 0:
                    inside = True
                    break
            if not inside:
                for sys_b in accepted_systems:
                    if vertical_overlap_ratio(s["box"], sys_b["box"]) > 0.60 and horizontal_overlap_or_gap(s["box"], sys_b["box"]) > 0:
                        inside = True
                        break
            if not inside:
                staves_outside.append(s)

        accepted_staves = heal_collinear_segments(staves_outside)
        for s in accepted_staves:
            s["class"] = "staff"

        # 4. Consolidate vertically adjacent single staves that share the same horizontal span into grand_staff
        all_staves = sorted(accepted_staves, key=lambda item: item["box"][1])
        consolidated_staves = []
        skip_indices = set()
        for i in range(len(all_staves)):
            if i in skip_indices:
                continue
            c1 = all_staves[i]
            merged_as_grand = False
            if i + 1 < len(all_staves):
                c2 = all_staves[i + 1]
                x1_max = max(c1["box"][0], c2["box"][0])
                x2_min = min(c1["box"][2], c2["box"][2])
                w1 = c1["box"][2] - c1["box"][0]
                w2 = c2["box"][2] - c2["box"][0]
                h_overlap = (x2_min - x1_max) / float(min(w1, w2)) if x2_min > x1_max else 0.0
                v_gap = c2["box"][1] - c1["box"][3]
                total_h = c2["box"][3] - c1["box"][1]

                # Piano grand staff constraint: gap <= 55 px, total height <= 170 px
                if h_overlap >= 0.70 and 0 <= v_gap <= 55 and total_h <= 170:
                    merged_box = [
                        min(c1["box"][0], c2["box"][0]),
                        c1["box"][1],
                        max(c1["box"][2], c2["box"][2]),
                        c2["box"][3]
                    ]
                    accepted_grand.append({
                        "class": "grand_staff",
                        "confidence": max(c1["confidence"], c2["confidence"]),
                        "box": merged_box
                    })
                    skip_indices.add(i + 1)
                    merged_as_grand = True
            if not merged_as_grand:
                consolidated_staves.append(c1)

        selected_candidates = accepted_grand + consolidated_staves + accepted_systems

        # Physical staff line verification (filters analysis brackets, slurs, divider lines)
        valid_candidates = []
        for cand in selected_candidates:
            x1, y1, x2, y2 = cand["box"]
            patch = img_bgr[y1:y2, x1:x2]
            if is_valid_music_staff(patch, cand["class"], cand["confidence"]):
                valid_candidates.append(cand)

        # Vertical collision suppression for duplicate/nested detections
        deduped = []
        for i, c1 in enumerate(valid_candidates):
            suppress = False
            for j, c2 in enumerate(valid_candidates):
                if i != j:
                    v_ratio = vertical_overlap_ratio(c1["box"], c2["box"])
                    h_rel = horizontal_overlap_or_gap(c1["box"], c2["box"])
                    if v_ratio > 0.35 and h_rel > 0:
                        # If a noisy system overlaps with a genuine grand staff, suppress the system
                        if c1["class"] == "system" and c2["class"] == "grand_staff":
                            if c1["confidence"] < 0.70:
                                suppress = True
                                break
                        elif c2["confidence"] > c1["confidence"] + 0.10:
                            suppress = True
                            break
                        elif abs(c2["confidence"] - c1["confidence"]) <= 0.10:
                            w1 = c1["box"][2] - c1["box"][0]
                            w2 = c2["box"][2] - c2["box"][0]
                            if (w2 * (c2["box"][3] - c2["box"][1])) > (w1 * (c1["box"][3] - c1["box"][1])):
                                suppress = True
                                break
            if not suppress:
                deduped.append(c1)

        selected_candidates = deduped

        # Sort candidates top-to-bottom by raw y1
        selected_candidates.sort(key=lambda item: item["box"][1])

        filtered = []
        num_cands = len(selected_candidates)

        for idx_cand, d in enumerate(selected_candidates):
            cls_name = d["class"]
            x1, y1, x2, y2 = d["box"]
            bw = x2 - x1
            bh = y2 - y1

            # Estimate staff space S for this specific candidate (fully scale-adaptive)
            cand_crop = img_bgr[y1:y2, x1:x2]
            s_est, _ = estimate_staff_spacing(cand_crop, cls_name)

            # Snap to outermost authentic horizontal staff lines
            strip = gray[y1:y2, x1:x2]
            if strip.size > 0:
                bg_val = float(np.percentile(strip, 90))
                k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                strip_bin = (strip < bg_val - 40).astype(np.uint8) * 255
                strip_lines = cv2.morphologyEx(strip_bin, cv2.MORPH_OPEN, k_line)
                row_ink = np.sum(strip_lines > 0, axis=1)
                line_rows = np.where(row_ink > int(bw * 0.15))[0]
                if len(line_rows) > 0:
                    top_staff_y = y1 + int(line_rows[0])
                    bot_staff_y = y1 + int(line_rows[-1])
                else:
                    top_staff_y = y1
                    bot_staff_y = y2
            else:
                top_staff_y = y1
                bot_staff_y = y2

            # Bounded padding: capture ledger lines, note stems, dynamics, while preserving text headings
            pad_y = int(min(24, max(10, s_est * 1.5)))
            pad_x = int(min(35, max(12, s_est * 2.0)))

            px1 = max(0, x1 - pad_x)
            px2 = min(img_w, x2 + pad_x)

            prev_y2 = filtered[-1]["padded_box"][3] if filtered else None
            py1 = max(0, top_staff_y - pad_y)
            if prev_y2 is not None and prev_y2 < top_staff_y:
                py1 = max(py1, prev_y2 + 4)

            next_y1 = selected_candidates[idx_cand + 1]["box"][1] if idx_cand + 1 < num_cands else None
            py2 = min(img_h, bot_staff_y + pad_y)
            if next_y1 is not None and next_y1 > bot_staff_y:
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
