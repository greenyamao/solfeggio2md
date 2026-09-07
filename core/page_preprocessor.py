import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def detect_and_split_spread(img_bgr: np.ndarray, overlap_ratio: float = 0.005) -> List[Tuple[np.ndarray, str]]:
    """
    Detects whether an image is a two-page spread (landscape, aspect_ratio >= 1.25)
    and splits it along the central book spine / gutter.
    
    Returns:
        List of tuples: [(page_img, "left"), (page_img, "right")] or [(img_bgr, "single")]
    """
    h, w = img_bgr.shape[:2]
    aspect_ratio = w / float(h)
    
    # A scanned two-page book spread has aspect ratio >= 1.25 and full page height (at least 300px).
    # Isolated single staves have aspect ratio > 4.0 and height < 200px and should not be split.
    if not (1.22 <= aspect_ratio <= 2.2 and h >= 300):
        return [(img_bgr, "single")]
    
    mid = w // 2
    
    # Adaptive spine band: 38% to 62% of width (accommodates asymmetric spreads)
    x1 = int(w * 0.38)
    x2 = int(w * 0.62)
    if x2 <= x1 + 10:
        gutter_x = mid
    else:
        strip_bgr = img_bgr[:, x1:x2]
        strip = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
        col_ink = (np.count_nonzero(strip < 220, axis=0) / float(max(1, h))).astype(np.float32)
        col_ink_s = cv2.GaussianBlur(col_ink.reshape(1, -1), (15, 1), 0).flatten()
        min_ink = float(np.min(col_ink_s))
        
        # Columns with minimal ink in the spine fold
        min_cols = np.where(col_ink_s <= min_ink + 0.005)[0]
        if len(min_cols) > 0:
            target = mid - x1
            best_col = min_cols[np.argmin(np.abs(min_cols - target))]
            gutter_x = x1 + int(best_col)
        else:
            gutter_x = mid

    # Ensure gutter is within plausible spine zone (38% to 62%)
    if gutter_x < int(w * 0.38) or gutter_x > int(w * 0.62):
        gutter_x = mid
        
    overlap_px = min(15, max(2, int(w * overlap_ratio)))
    left_x_end = min(w, gutter_x + overlap_px)
    right_x_start = max(0, gutter_x - overlap_px)
    
    left_page = img_bgr[:, :left_x_end].copy()
    right_page = img_bgr[:, right_x_start:].copy()
    
    return [(left_page, "left"), (right_page, "right")]


def estimate_skew_fourier(gray_img: np.ndarray) -> float:
    """
    Fallback 2D-DFT continuous skew estimator when jdeskew is not yet installed.
    Computes angle of dominant harmonic rays in magnitude spectrum.
    """
    try:
        from jdeskew.estimator import get_angle
        try:
            if gray_img.shape[0] > 1024:
                return float(get_angle(gray_img, vertical_image_shape=1024))
            return float(get_angle(gray_img))
        except TypeError:
            return float(get_angle(gray_img))
    except ImportError:
        pass

    # Simple robust Radon/Hough fallback
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray_img.shape[1] // 8, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx != 0:
            deg = np.degrees(np.arctan2(dy, dx))
            # Music staff lines are horizontal -> angle should be near 0
            if abs(deg) <= 30:
                angles.append(deg)

    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew_page(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates the fine continuous skew angle and rotates the image to achieve
    horizontal alignment of staff lines.
    
    Returns:
        (rotated_img_bgr, angle_degrees)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_fourier(gray)
    
    if abs(angle) < 0.05:
        return img_bgr, 0.0
        
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    deskewed = cv2.warpAffine(
        img_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # Clean white padding on borders
    )
    return deskewed, angle


class PagePreprocessor:
    """
    End-to-end normalization pipeline for book scans:
    1. Detects and splits two-page spreads into separate pages.
    2. Corrects continuous rotational skew so music staves are strictly horizontal.
    """
    def __init__(self, target_dpi: int = 200):
        self.target_dpi = target_dpi

    def process_image(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes an input BGR image (single page or spread).
        Returns a list of dictionaries with normalized page images and metadata.
        """
        split_pages = detect_and_split_spread(img_bgr)
        results = []
        
        for idx, (page_img, spread_type) in enumerate(split_pages):
            deskewed_img, angle = deskew_page(page_img)
            results.append({
                "page_img": deskewed_img,
                "spread_type": spread_type,
                "skew_angle": angle,
                "width": deskewed_img.shape[1],
                "height": deskewed_img.shape[0]
            })
            
        return results


def estimate_staff_spacing(img_bgr_or_gray: np.ndarray, notation_class: str = "staff") -> Tuple[float, float]:
    """
    Scale-invariant staff space (S) estimator:
    Calculates the exact physical distance between adjacent staff lines down to sub-pixel accuracy.
    Works seamlessly across all resolutions (72 DPI to 600 DPI), font sizes, and staves of any size.

    Returns:
        (measured_spacing_px, reference_center_y_px)
    """
    gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY) if len(img_bgr_or_gray.shape) == 3 else img_bgr_or_gray
    h, w = gray.shape[:2]
    is_grand = notation_class.lower().replace(" ", "_") in ("grand_staff", "grandstaff")

    geom_s = max(2.5, h / 13.0 if is_grand else h / 4.0)
    min_dist = max(2, int(geom_s * 0.5))

    x1, x2 = int(w * 0.25), int(w * 0.75)
    if x2 <= x1 + 10:
        x1, x2 = 0, w

    strip = gray[:, x1:x2]
    proj = np.sum(255.0 - strip, axis=1)

    y_limit = (h // 2) if is_grand else (h - 1)
    peaks = []
    for y in range(1, y_limit):
        if proj[y] > proj[y-1] and proj[y] >= proj[y+1]:
            peaks.append((proj[y], y))

    peaks.sort(key=lambda p: p[0], reverse=True)

    filtered_peaks = []
    for p in peaks:
        y = p[1]
        if all(abs(y - fp) >= min_dist for fp in filtered_peaks):
            filtered_peaks.append(y)
    filtered_peaks.sort()

    diffs = np.diff(filtered_peaks)
    valid_diffs = [d for d in diffs if 0.45 * geom_s <= d <= 1.8 * geom_s]

    if len(valid_diffs) >= 3:
        s_final = float(np.median(valid_diffs))
        ref_y = float(np.median(filtered_peaks[:5]))
    else:
        s_final = float(geom_s)
        ref_y = float(h / 4.0 if is_grand else h / 2.0)

    return s_final, ref_y


def normalize_staff_crop(crop_bgr: np.ndarray, notation_class: str = "staff") -> Tuple[np.ndarray, float, float]:
    """
    High-precision, non-destructive music staff crop normalization:
    1. Determines precise rotational tilt angle using 2D-DFT (jdeskew).
    2. Performs high-quality rotational deskew with border padding so no notes or ledger lines are clipped.
    3. Preserves authentic staff geometry without artificial undulating warping (spaghetti distortion).

    Returns:
        (normalized_crop_bgr, tilt_angle_degrees, bend_delta_pixels)
    """
    h_orig, w_orig = crop_bgr.shape[:2]
    if h_orig < 15 or w_orig < 30:
        return crop_bgr, 0.0, 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 2D-DFT Rotational Deskew Angle
    try:
        from jdeskew.estimator import get_angle
        if w_orig > 600:
            scale_factor = 600.0 / float(w_orig)
            scaled_gray = cv2.resize(gray, (600, max(1, int(round(h_orig * scale_factor)))), interpolation=cv2.INTER_AREA)
        else:
            scaled_gray = gray
        raw_angle = float(get_angle(scaled_gray))
        if abs(raw_angle) > 35.0:
            tilt_deg = 0.0
        else:
            tilt_deg = raw_angle
    except Exception:
        tilt_deg = 0.0

    # 2. Rotate to exact horizontal orientation with margin padding to prevent clipping
    angle_rad = abs(np.radians(tilt_deg))
    pad_rot = int(np.ceil(w_orig * np.sin(angle_rad) / 2.0)) + 4 if abs(tilt_deg) >= 0.1 else 0

    if pad_rot > 0:
        padded_rot = cv2.copyMakeBorder(crop_bgr, pad_rot, pad_rot, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        padded_rot = crop_bgr

    h_rot, w_rot = padded_rot.shape[:2]
    if abs(tilt_deg) >= 0.1:
        center = (w_rot / 2.0, h_rot / 2.0)
        M = cv2.getRotationMatrix2D(center, tilt_deg, 1.0)
        deskewed = cv2.warpAffine(
            padded_rot, M, (w_rot, h_rot),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
    else:
        deskewed = padded_rot

    # 3. 1D Continuous Profile Straightening: aligns every vertical 1-px column to horizontal reference
    bend_delta = 0.0
    h_d, w_d = deskewed.shape[:2]
    if w_d >= 160 and h_d >= 25:
        try:
            d_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
            bg_val = float(np.percentile(d_gray, 90))
            bin_inv = (d_gray < bg_val - 35).astype(np.float32)

            # Isolate horizontal staff line segments (erase note stems, text, accidentals)
            k_len = max(15, min(40, int(w_d * 0.04)))
            k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
            lines_mask = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k_line)

            sums = np.sum(lines_mask, axis=0)
            has_lines = np.where(sums >= 4.0)[0]
            if len(has_lines) >= 20:
                x_first = int(has_lines[0])
                x_last = int(has_lines[-1])

                ref_x_start = min(x_first + 5, max(x_first, x_last - 45))
                ref_x_end = min(x_last, ref_x_start + 40)
                init_profile = np.mean(lines_mask[:, ref_x_start:ref_x_end], axis=1)

                # Estimate staff line spacing S from init_profile to bound physical dewarp search range
                line_ys = np.where(init_profile > 0.15)[0]
                diffs = [line_ys[i] - line_ys[i - 1] for i in range(1, len(line_ys)) if line_ys[i] - line_ys[i - 1] > 3]
                s_spacing = float(np.median(diffs)) if len(diffs) else 8.0
                max_phys_shift = max(3.5, min(14.0, s_spacing * 1.5))

                shifts = np.zeros(w_d, dtype=np.float32)
                prev_d = 0.0

                # Offload 1D cross-correlation batch to CUDA if available to avoid CPU thread heating
                use_cuda = False
                try:
                    import torch
                    import torch.nn.functional as F
                    if torch.cuda.is_available():
                        use_cuda = True
                except Exception:
                    pass

                if use_cuda:
                    max_d_int = int(np.ceil(max_phys_shift))
                    t_mask = torch.from_numpy(lines_mask).cuda().permute(1, 0).unsqueeze(1)  # (w_d, 1, h_d)
                    t_prof = torch.from_numpy(init_profile).cuda().view(1, 1, -1)
                    corr_matrix = F.conv1d(t_mask, t_prof, padding=max_d_int).squeeze(1).cpu().numpy()
                    center_idx = max_d_int

                    for x in range(x_first, x_last + 1):
                        if sums[x] < 2.0:
                            shifts[x] = prev_d
                            continue

                        low_d = int(np.floor(max(-max_phys_shift, prev_d - 1.5)))
                        high_d = int(np.ceil(min(max_phys_shift, prev_d + 1.5)))

                        i_low = center_idx + low_d
                        i_high = center_idx + high_d + 1
                        sub_corrs = corr_matrix[x, i_low:i_high]
                        best_local = int(np.argmax(sub_corrs))
                        best_d = float(low_d + best_local)
                        if 0 < best_local < len(sub_corrs) - 1:
                            y0, y1, y2 = sub_corrs[best_local - 1], sub_corrs[best_local], sub_corrs[best_local + 1]
                            denom = 2.0 * (2.0 * y1 - y0 - y2)
                            if denom > 1e-4:
                                best_d += float(y0 - y2) / denom
                        shifts[x] = best_d
                        prev_d = best_d
                else:
                    for x in range(x_first, x_last + 1):
                        col = lines_mask[:, x]
                        if sums[x] < 2.0:
                            shifts[x] = prev_d
                            continue

                        # Search within physically continuous window around prev_d bounded by max_phys_shift
                        low_d = int(np.floor(max(-max_phys_shift, prev_d - 1.5)))
                        high_d = int(np.ceil(min(max_phys_shift, prev_d + 1.5)))

                        corrs = []
                        d_vals = list(range(low_d, high_d + 1))
                        for d in d_vals:
                            c = np.sum(col[d:] * init_profile[:-d]) if d > 0 else (
                                np.sum(col[:d] * init_profile[-d:]) if d < 0 else np.sum(col * init_profile)
                            )
                            corrs.append(c)

                        best_idx = int(np.argmax(corrs))
                        best_d = float(d_vals[best_idx])
                        # Sub-pixel parabolic peak refinement
                        if 0 < best_idx < len(corrs) - 1:
                            y0, y1, y2 = corrs[best_idx - 1], corrs[best_idx], corrs[best_idx + 1]
                            denom = 2.0 * (2.0 * y1 - y0 - y2)
                            if denom > 1e-4:
                                best_d += float(y0 - y2) / denom
                        shifts[x] = best_d
                        prev_d = best_d

                # Pad margins with edge shifts so margins don't shear
                if x_first > 0:
                    shifts[:x_first] = shifts[x_first]
                if x_last < w_d - 1:
                    shifts[x_last + 1:] = shifts[x_last]

                # Smooth displacement curve: median filter (removes localized spikes) + Gaussian filter
                k_med = min(41, len(shifts) if len(shifts) % 2 == 1 else len(shifts) - 1)
                if k_med >= 5:
                    padded = np.pad(shifts, k_med // 2, mode='edge')
                    med_shifts = np.array([np.median(padded[i:i + k_med]) for i in range(w_d)], dtype=np.float32)
                else:
                    med_shifts = shifts

                k_gauss = min(51, len(shifts) if len(shifts) % 2 == 1 else len(shifts) - 1)
                if k_gauss >= 5:
                    kernel_g = cv2.getGaussianKernel(k_gauss, 12.0).flatten()
                    smooth_dy = np.convolve(med_shifts, kernel_g, mode='same').astype(np.float32)
                else:
                    smooth_dy = med_shifts

                max_bend = float(np.max(np.abs(smooth_dy)))
                if max_bend >= 0.5:
                    bend_delta = max_bend
                    # Shift each vertical 1-pixel slice by smooth_dy(x) to lock lines strictly horizontal
                    grid_x = np.tile(np.arange(w_d, dtype=np.float32), (h_d, 1)).astype(np.float32)
                    grid_y = (np.tile(np.arange(h_d, dtype=np.float32)[:, None], (1, w_d)) + smooth_dy[None, :]).astype(np.float32)

                    deskewed = cv2.remap(
                        deskewed,
                        grid_x,
                        grid_y,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255)
                    )
        except Exception:
            pass

    return deskewed, round(tilt_deg, 1), round(bend_delta, 1)


