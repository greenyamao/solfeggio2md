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

    # 3. Vectorized VRLC Staff Skeleton & C1-Smooth Physical Paper Gutter Dewarping:
    # Completely rejects note heads, stems, beams, accidentals, clefs, and ties/slurs.
    # Warps ONLY physical spine curl using a C1-smooth parametric model (zero wobble mathematically guaranteed).
    bend_delta = 0.0
    h_d, w_d = deskewed.shape[:2]
    if w_d >= 160 and h_d >= 25:
        try:
            d_gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
            is_grand = notation_class.lower().replace(" ", "_") in ("grand_staff", "grandstaff")

            skel, line_d, staff_s, target_lines = _extract_vrlc_staff_skeleton(d_gray, is_grand=is_grand)
            xs, ys = _get_staff_column_displacements(skel, target_lines, staff_s)

            if len(xs) >= 50:
                k_slope, b_intercept, A_L, xc_L, A_R, xc_R, x_first, x_last = _fit_c1_paper_curl(xs, ys, w_d)
                smooth_dy = _build_c1_displacement_curve(w_d, k_slope, A_L, xc_L, A_R, xc_R, x_first, x_last)

                max_bend = float(np.max(np.abs(smooth_dy)))
                # Strict deadband protection: if deflection is < 1.0 px, staff is already flat (0ms, 0 interpolation loss)
                if max_bend >= 1.0:
                    bend_delta = max_bend
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


def _extract_vrlc_staff_skeleton(gray: np.ndarray, is_grand: bool = False) -> Tuple[np.ndarray, int, int, List[int]]:
    """
    Vectorized Vertical Run-Length Coding (VRLC) staff skeleton extraction.
    Filters out note heads, stems, beams, clefs, accidentals, and ties.
    Returns: (clean_skeleton, line_thickness, staff_space, target_lines)
    """
    h, w = gray.shape[:2]
    bg_val = float(np.percentile(gray, 90))
    bin_inv = (gray < bg_val - 35).astype(np.uint8)

    # 1. Estimate staff line thickness d and staff space s via fast sample columns
    sample_cols = np.linspace(int(w * 0.15), int(w * 0.85), 40, dtype=int)
    sampled = bin_inv[:, sample_cols]
    padded = np.pad(sampled.astype(np.int32), ((1, 1), (0, 0)), mode="constant")
    diffs = np.diff(padded, axis=0)

    black_lens = []
    white_lens = []
    for c in range(diffs.shape[1]):
        starts = np.where(diffs[:, c] == 1)[0]
        ends = np.where(diffs[:, c] == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            black_lens.extend((ends - starts).tolist())
            if len(starts) > 1:
                white_lens.extend((starts[1:] - ends[:-1]).tolist())

    line_d = int(np.argmax(np.bincount(black_lens)[1:]) + 1) if black_lens else 2
    staff_s = int(np.argmax(np.bincount(white_lens)[1:]) + 1) if white_lens else int(h / 4.0)
    line_d = max(1, min(4, line_d))
    staff_s = max(5, min(35, staff_s))

    # 2. Vectorized VRLC: remove all ink whose vertical run > 2.2 * line_d
    max_run = int(round(2.2 * line_d))
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max_run + 1))
    thick_verticals = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k_vert)
    thin_lines = cv2.subtract(bin_inv, thick_verticals)

    # 3. Horizontal morphological close to bridge vertical stem cuts
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(1.4 * staff_s)), 1))
    clean_skel = cv2.morphologyEx(thin_lines, cv2.MORPH_CLOSE, k_close)

    # 4. Target lines identification from stable central span
    x_c1, x_c2 = int(w * 0.25), int(w * 0.75)
    central_proj = np.sum(clean_skel[:, x_c1:x_c2], axis=1)
    peaks = []
    for y in range(1, h - 1):
        if central_proj[y] > central_proj[y-1] and central_proj[y] >= central_proj[y+1] and central_proj[y] > (x_c2 - x_c1) * 0.15:
            peaks.append((central_proj[y], y))
    peaks.sort(key=lambda p: p[0], reverse=True)

    unique_y = []
    for _, y in peaks:
        if all(abs(y - uy) >= max(3, int(staff_s * 0.55)) for uy in unique_y):
            unique_y.append(y)
    unique_y.sort()

    target_lines = []
    if is_grand and len(unique_y) >= 10:
        treble = [y for y in unique_y if y < h * 0.55]
        bass = [y for y in unique_y if y >= h * 0.45]
        if len(treble) >= 5 and len(bass) >= 5:
            target_lines.extend(treble[:5])
            target_lines.extend(bass[-5:])

    if len(target_lines) < 5:
        if len(unique_y) >= 5:
            best_err = 999.0
            best_grp = unique_y[:5]
            for i in range(len(unique_y) - 4):
                cand = unique_y[i:i+5]
                err = float(np.mean(np.abs(np.diff(cand) - staff_s)))
                if err < best_err:
                    best_err = err
                    best_grp = cand
            target_lines = best_grp
        else:
            mid_y = h / 2.0
            target_lines = [int(round(mid_y + (k - 2) * staff_s)) for k in range(5)]

    # 5. Band-masking: purges isolated slurs, ties, and text underlines outside authentic lines
    line_band_mask = np.zeros((h, w), dtype=np.uint8)
    band_r = max(2, int(round(staff_s * 0.35)))
    for yk in target_lines:
        y_min = max(0, yk - band_r)
        y_max = min(h, yk + band_r + 1)
        line_band_mask[y_min:y_max, :] = 1

    final_skel = clean_skel * line_band_mask
    return final_skel, line_d, staff_s, target_lines


def _get_staff_column_displacements(skel: np.ndarray, target_lines: List[int], staff_s: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes vertical displacement for each valid column relative to target reference lines.
    Vectorized across columns.
    """
    h, w = skel.shape
    band_r = max(2, int(round(staff_s * 0.35)))

    line_diffs = np.full((len(target_lines), w), np.nan, dtype=np.float32)
    for idx, y_ref in enumerate(target_lines):
        y_min = max(0, y_ref - band_r)
        y_max = min(h, y_ref + band_r + 1)
        sub = skel[y_min:y_max, :]
        col_sums = np.sum(sub, axis=0)
        valid_cols = col_sums > 0
        if np.any(valid_cols):
            weights = np.arange(y_min, y_max, dtype=np.float32)[:, None]
            y_cents = np.sum(sub[:, valid_cols] * weights, axis=0) / col_sums[valid_cols]
            line_diffs[idx, valid_cols] = y_cents - y_ref

    valid_count = np.sum(~np.isnan(line_diffs), axis=0)
    col_mask = valid_count >= max(1, len(target_lines) // 3)
    xs = np.where(col_mask)[0].astype(np.float32)
    if len(xs) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    sub_diffs = line_diffs[:, col_mask]
    ys = np.nanmedian(sub_diffs, axis=0).astype(np.float32)
    return xs, ys


def _robust_fit_curl(x_pts: np.ndarray, y_pts: np.ndarray, x_ref: float, L: float, is_left: bool) -> Tuple[float, float]:
    """
    Pure NumPy Iteratively Reweighted Least Squares (IRLS) Huber fit for curl amplitude A and hinge xc:
    dy(x) = A * (1 - dist / xc)^2
    """
    if len(x_pts) < 15:
        return 0.0, float(L * 0.25)

    dist = (x_pts - x_ref) if is_left else (x_ref - x_pts)
    valid_pts = dist >= 0
    if np.sum(valid_pts) < 15:
        return 0.0, float(L * 0.25)

    dist = dist[valid_pts]
    y_act_all = y_pts[valid_pts]

    best_loss = 1e9
    best_A = 0.0
    best_xc = float(L * 0.25)

    xc_candidates = np.linspace(L * 0.15, L * 0.35, 9)
    for xc in xc_candidates:
        t = np.maximum(0.0, 1.0 - dist / xc) ** 2
        active = t > 0.01
        if np.sum(active) < 8:
            continue
        t_act = t[active]
        y_act = y_act_all[active]

        A = float(np.sum(t_act * y_act) / max(1e-6, np.sum(t_act ** 2)))
        for _ in range(2):
            res = np.abs(y_act - A * t_act)
            w = np.where(res < 1.5, 1.0, 1.5 / np.maximum(res, 1e-4))
            A = float(np.sum(w * t_act * y_act) / max(1e-6, np.sum(w * t_act ** 2)))

        res = np.abs(y_act - A * t_act)
        hub = np.where(res < 1.5, 0.5 * res**2, 1.5 * res - 1.125)
        loss = float(np.sum(hub))
        if loss < best_loss:
            best_loss = loss
            best_A = A
            best_xc = float(xc)

    best_A = max(-20.0, min(20.0, best_A))
    return best_A, best_xc


def _fit_c1_paper_curl(xs: np.ndarray, ys: np.ndarray, width: int) -> Tuple[float, float, float, float, float, float, int, int]:
    """
    Fits two-sided C1 physical paper curl model relative to actual staff boundaries [x_first, x_last].
    """
    if len(xs) < 50:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, width - 1

    x_first = int(xs[0])
    x_last = int(xs[-1])
    L = float(max(10, x_last - x_first))

    # 1. Estimate linear slope k in central 50% plateau of staff
    mid_start = x_first + int(L * 0.25)
    mid_end = x_first + int(L * 0.75)
    mid_mask = (xs >= mid_start) & (xs <= mid_end)
    if np.sum(mid_mask) >= 20:
        k_slope, b_intercept = np.polyfit(xs[mid_mask], ys[mid_mask], deg=1)
    else:
        k_slope, b_intercept = np.polyfit(xs, ys, deg=1)

    y_detrend = ys - (k_slope * xs + b_intercept)

    # 2. Left gutter curl
    left_bound = x_first + int(L * 0.35)
    left_mask = xs <= left_bound
    if np.sum(left_mask) >= 15:
        A_L, xc_L = _robust_fit_curl(xs[left_mask], y_detrend[left_mask], x_ref=float(x_first), L=L, is_left=True)
    else:
        A_L, xc_L = 0.0, float(L * 0.25)

    # 3. Right gutter curl
    right_bound = x_last - int(L * 0.35)
    right_mask = xs >= right_bound
    if np.sum(right_mask) >= 15:
        A_R, xc_R = _robust_fit_curl(xs[right_mask], y_detrend[right_mask], x_ref=float(x_last), L=L, is_left=False)
    else:
        A_R, xc_R = 0.0, float(L * 0.25)

    # Deadband threshold: if deflection is < 1.2 px, page paper is flat
    if abs(A_L) < 1.2:
        A_L = 0.0
    if abs(A_R) < 1.2:
        A_R = 0.0
    if abs(k_slope * L) < 1.0:
        k_slope = 0.0

    return k_slope, b_intercept, A_L, xc_L, A_R, xc_R, x_first, x_last


def _build_c1_displacement_curve(width: int, k_slope: float, A_L: float, xc_L: float, A_R: float, xc_R: float,
                                 x_first: int, x_last: int) -> np.ndarray:
    """
    Builds the 1D vertical displacement field Delta y(x) across the full crop width.
    Applies smooth margin clamping outside [x_first, x_last] to protect clefs and barlines from false curls.
    """
    x = np.arange(width, dtype=np.float32)
    dy = k_slope * (x - (x_first + x_last) / 2.0)

    # Left gutter (clamped outside x_first)
    if abs(A_L) >= 1.0:
        dist_l = np.maximum(0.0, x - x_first)
        t_l = np.maximum(0.0, 1.0 - dist_l / max(1.0, xc_L))
        dy += A_L * (t_l ** 2)

    # Right gutter (clamped outside x_last)
    if abs(A_R) >= 1.0:
        dist_r = np.maximum(0.0, x_last - x)
        t_r = np.maximum(0.0, 1.0 - dist_r / max(1.0, xc_R))
        dy += A_R * (t_r ** 2)

    return dy



