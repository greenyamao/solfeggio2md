from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


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

    def detect(self, img_bgr: np.ndarray, imgsz: int = 1024) -> List[Dict[str, Any]]:
        """
        Runs layout detection on an image.
        Returns a list of structured detection dictionaries.
        """
        self._ensure_loaded()
        img_h, img_w = img_bgr.shape[:2]
        
        # Inference with Ultralytics YOLO
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
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
        
        # Filter and prioritize detections:
        # If a single 'staff' is physically inside an identified 'grand_staff',
        # suppress the child single 'staff' to avoid double-cropping.
        # Group raw detections by canonical class
        # OLA v2 classes: 0: system_measures, 1: stave_measures, 2: staves, 3: systems, 4: grand_staff
        raw_grand = []
        raw_staves = []
        raw_systems = []
        
        for d in raw_detections:
            cls_name = d["class"].lower().replace(" ", "_")
            if cls_name in ("grand_staff", "grandstaff"):
                raw_grand.append(d)
            elif cls_name in ("staves", "staff"):
                raw_staves.append(d)
            elif cls_name in ("systems", "system"):
                raw_systems.append(d)

        def box_iou(b1, b2):
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

        def vertical_overlap_ratio(b1, b2):
            """Fraction of the smaller box's height that is vertically overlapped."""
            y1 = max(b1[1], b2[1])
            y2 = min(b1[3], b2[3])
            if y2 <= y1:
                return 0.0
            h1 = b1[3] - b1[1]
            h2 = b2[3] - b2[1]
            return (y2 - y1) / float(max(1, min(h1, h2)))

        def horizontal_overlap_or_gap(b1, b2):
            """Returns positive value for overlap, negative value for gap."""
            x1 = max(b1[0], b2[0])
            x2 = min(b1[2], b2[2])
            if x2 > x1:
                return x2 - x1  # positive overlap
            # gap: distance between right of one and left of the other
            if b1[2] <= b2[0]:
                return -(b2[0] - b1[2])
            else:
                return -(b1[0] - b2[2])

        def merge_boxes(b1, b2):
            return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

        def heal_collinear_segments(detections_list, v_overlap_thresh=0.65, max_gap_px=25):
            """
            Merges horizontally broken or overlapping segments of the same staff/grand_staff line.
            """
            if not detections_list:
                return []
            
            merged = []
            # Sort by confidence descending
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
                    v_ratio = vertical_overlap_ratio(cur_box, other_box)
                    h_rel = horizontal_overlap_or_gap(cur_box, other_box)

                    # If on the same horizontal staff band and overlapping horizontally or adjacent
                    if v_ratio >= v_overlap_thresh and h_rel >= -max_gap_px:
                        cur_box = merge_boxes(cur_box, other_box)
                        cur_conf = max(cur_conf, sorted_dets[j]["confidence"])
                        used[j] = True

                used[i] = True
                merged.append({
                    "class": cur_cls,
                    "confidence": cur_conf,
                    "box": cur_box
                })
            return merged

        # 1. Process grand_staff (highest priority: piano 2-staff systems)
        accepted_grand = heal_collinear_segments(raw_grand)
        # Standardize class name
        for g in accepted_grand:
            g["class"] = "grand_staff"

        # 2. Process staves: remove any staff that falls inside a grand_staff
        staves_outside_grand = []
        for s in raw_staves:
            # Check if vertically inside any grand_staff
            inside_grand = False
            for g in accepted_grand:
                v_ratio = vertical_overlap_ratio(s["box"], g["box"])
                h_rel = horizontal_overlap_or_gap(s["box"], g["box"])
                if v_ratio > 0.60 and h_rel > 0:
                    inside_grand = True
                    break
            if not inside_grand:
                staves_outside_grand.append(s)

        accepted_staves = heal_collinear_segments(staves_outside_grand)
        for s in accepted_staves:
            s["class"] = "staff"

        # 3. Process systems:
        # In OLA, systems often duplicates single staves (1 staff = 1 system) or overlaps grand_staff.
        # We only keep a system if it is NOT covered by grand_staff and NOT covered by accepted_staves.
        # If it matches a single staff (height is staff-sized), we ensure it is represented as a staff.
        accepted_systems = []
        for sys_det in raw_systems:
            # Check grand_staff collision
            if any(vertical_overlap_ratio(sys_det["box"], g["box"]) > 0.60 and horizontal_overlap_or_gap(sys_det["box"], g["box"]) > 0 for g in accepted_grand):
                continue
            # Check staff collision
            collides_with_staff = False
            for st in accepted_staves:
                if vertical_overlap_ratio(sys_det["box"], st["box"]) > 0.60 and horizontal_overlap_or_gap(sys_det["box"], st["box"]) > 0:
                    collides_with_staff = True
                    break
            if not collides_with_staff:
                # System is a multi-staff ensemble/choir or an unclassified staff
                sys_h = sys_det["box"][3] - sys_det["box"][1]
                # If height is small (< 85 px at 200 dpi), it's likely an isolated single staff
                if sys_h < 85:
                    accepted_staves.append({
                        "class": "staff",
                        "confidence": sys_det["confidence"],
                        "box": sys_det["box"]
                    })
                else:
                    accepted_systems.append({
                        "class": "system",
                        "confidence": sys_det["confidence"],
                        "box": sys_det["box"]
                    })

        # Final re-heal staves just in case an unclassified system merged with a staff
        accepted_staves = heal_collinear_segments(accepted_staves)
        accepted_systems = heal_collinear_segments(accepted_systems)

        selected_candidates = accepted_grand + accepted_staves + accepted_systems

        # Convert to grayscale for ink analysis
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Sort candidates top-to-bottom by raw y1
        selected_candidates.sort(key=lambda item: item["box"][1])

        filtered = []
        num_cands = len(selected_candidates)

        for idx_cand, d in enumerate(selected_candidates):
            cls_name = d["class"]
            x1, y1, x2, y2 = d["box"]
            bw = x2 - x1
            bh = y2 - y1

            # Staff spacing S estimate (distance between 2 adjacent staff lines)
            if cls_name == "grand_staff":
                s_est = max(8, min(18, bh // 10))
            else:
                s_est = max(8, min(18, bh // 4))

            # Horizontal expansion:
            # Grand staff needs room for accolade/brace on the left (at least 38 px)
            # Single staff needs room for clef and key signature (at least 32 px)
            if cls_name == "grand_staff":
                pad_l = max(38, int(bw * 0.05) + 20)
                pad_r = max(25, int(bw * 0.03) + 15)
            else:
                pad_l = max(32, int(bw * 0.04) + 15)
                pad_r = max(25, int(bw * 0.03) + 15)

            px1 = max(0, x1 - pad_l)
            px2 = min(img_w, x2 + pad_r)

            # Vertical expansion via ink profile
            staff_strip = gray[y1:y2, x1:x2]
            bg_val = float(np.percentile(staff_strip, 90)) if staff_strip.size > 0 else 250.0
            ink_thresh = bg_val - 45.0

            inside_ink = np.sum(gray[y1:y2, px1:px2] < ink_thresh, axis=1)
            median_inside_ink = float(np.median(inside_ink)) if len(inside_ink) > 0 else 100.0
            noise_thresh = max(3, int(median_inside_ink * 0.03))

            # Upward expansion limit: don't cross midpoint to previous candidate
            prev_y2 = selected_candidates[idx_cand - 1]["box"][3] if idx_cand > 0 else None
            max_up = int(s_est * 5.0)
            if prev_y2 is not None and prev_y2 < y1:
                avail_up = max(8, (y1 - prev_y2) // 2)
                max_up = min(max_up, avail_up)
            max_up = min(max_up, y1)

            zero_run_up = 0
            best_up = 0
            for dy in range(1, max_up + 1):
                y_curr = y1 - dy
                row = gray[y_curr, px1:px2]
                cnt = int(np.sum(row < ink_thresh))
                if cnt > noise_thresh:
                    best_up = dy
                    zero_run_up = 0
                else:
                    zero_run_up += 1
                    if zero_run_up >= 8:
                        break

            # Place top margin safely in whitespace
            pad_t = max(int(s_est * 1.5), best_up + min(int(s_est * 1.0), 10))
            if prev_y2 is not None and prev_y2 < y1:
                pad_t = min(pad_t, max(6, y1 - prev_y2 - 6))
            pad_t = min(pad_t, y1)
            py1 = y1 - pad_t

            # Downward expansion limit: don't cross midpoint to next candidate
            next_y1 = selected_candidates[idx_cand + 1]["box"][1] if idx_cand + 1 < num_cands else None
            max_down = int(s_est * 5.0)
            if next_y1 is not None and next_y1 > y2:
                avail_down = max(8, (next_y1 - y2) // 2)
                max_down = min(max_down, avail_down)
            max_down = min(max_down, img_h - y2)

            zero_run_down = 0
            best_down = 0
            for dy in range(1, max_down + 1):
                y_curr = y2 + dy
                row = gray[y_curr, px1:px2]
                cnt = int(np.sum(row < ink_thresh))
                if cnt > noise_thresh:
                    best_down = dy
                    zero_run_down = 0
                else:
                    zero_run_down += 1
                    if zero_run_down >= 8:
                        break

            pad_b = max(int(s_est * 1.6), best_down + min(int(s_est * 1.0), 10))
            if next_y1 is not None and next_y1 > y2:
                pad_b = min(pad_b, max(6, next_y1 - y2 - 6))
            pad_b = min(pad_b, img_h - y2)
            py2 = y2 + pad_b

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
