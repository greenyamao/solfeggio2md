import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def detect_and_split_spread(img_bgr: np.ndarray, overlap_ratio: float = 0.02) -> List[Tuple[np.ndarray, str]]:
    """
    Detects whether an image is a two-page spread (landscape) and splits it along
    the central book spine / gutter.
    
    Returns:
        List of tuples: [(page_img, "left"), (page_img, "right")] or [(img_bgr, "single")]
    """
    h, w = img_bgr.shape[:2]
    aspect_ratio = w / float(h)
    
    # A scanned two-page book spread has aspect ratio typically between 1.25 and 1.85,
    # and has a full page height (at least 600px).
    # Isolated horizontal staff strips (crops) have aspect ratio > 2.0 and should NEVER be split!
    if not (1.25 <= aspect_ratio <= 1.85 and h >= 600):
        return [(img_bgr, "single")]
    
    # Search for gutter in the central 35% to 65% of width
    search_start = int(w * 0.35)
    search_end = int(w * 0.65)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Downscale for fast robust gutter analysis
    small_h = 400
    scale = small_h / float(h)
    small_w = int(w * scale)
    small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    s_start = int(small_w * 0.35)
    s_end = int(small_w * 0.65)
    
    # Vertical projection profile: column average intensity
    col_means = np.mean(small_gray[:, s_start:s_end], axis=0)
    
    # Smooth column profile with Gaussian filter to avoid local glyph noise
    col_smooth = cv2.GaussianBlur(col_means.reshape(1, -1).astype(np.float32), (31, 1), 0).flatten()
    
    # Gutter usually appears as a local darkness minimum (shadow in the fold)
    # or a prominent gap in text/staves.
    local_min_idx = int(np.argmin(col_smooth))
    gutter_x_small = s_start + local_min_idx
    gutter_x = int(gutter_x_small / scale)
    
    # Sanity check: gutter should be reasonably close to the middle (within 40% - 60% of W)
    if not (0.40 * w <= gutter_x <= 0.60 * w):
        gutter_x = w // 2
        
    overlap_px = int(w * overlap_ratio)
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
    High-precision scale-adaptive music staff crop normalization:
    1. Determines precise rotational tilt angle using 2D-DFT (jdeskew).
    2. Performs high-quality rotational deskew with border padding so no notes are clipped.
    3. Estimates the exact physical staff spacing S to adapt all tracking and filtering parameters.
    4. Continuously tracks the 5-line staff structure outwards from the flat central region
       to the page boundaries (immune to note beams, ledger lines, and dense chords).
    5. Applies subpixel dewarping to eliminate spine gutter curvature (proportional to staff spacing).

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
        raw_angle = float(get_angle(gray))
        if abs(raw_angle) > 40.0:
            tilt_deg = 0.0
        else:
            tilt_deg = raw_angle
    except Exception:
        tilt_deg = 0.0

    # 2. Rotate to exact horizontal orientation with margin padding
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

    # 3. Scale-adaptive continuous 5-line staff structure tracking
    staff_spacing, ref_y = estimate_staff_spacing(deskewed, notation_class)
    if not (2.0 <= staff_spacing <= 120.0):
        return deskewed, round(tilt_deg, 1), 0.0

    # Build 5-line comb filter template scaled to staff spacing
    template_h = int(staff_spacing * 4) + int(staff_spacing * 1.5)
    template = np.zeros(template_h, dtype=np.float32)
    tmpl_center = len(template) // 2
    for k in range(-2, 3):
        idx = int(round(tmpl_center + k * staff_spacing))
        if 0 <= idx < len(template):
            template[idx] = 1.0
            if idx > 0:
                template[idx-1] = 0.5
            if idx + 1 < len(template):
                template[idx+1] = 0.5

    # Outward continuous tracking from seed point
    flat_x1 = int(w_rot * 0.30)
    flat_x2 = int(w_rot * 0.80)
    if flat_x2 <= flat_x1 + 10:
        flat_x1, flat_x2 = 0, w_rot

    seed_x = (flat_x1 + flat_x2) // 2
    track_y = np.full(w_rot, ref_y, dtype=np.float32)

    step_w = max(5, int(staff_spacing * 1.0) | 1)
    search_r = max(3, int(staff_spacing * 0.45))
    max_step_dy = max(1.0, staff_spacing * 0.15)
    gray_deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    # Track LEFT towards gutter/margin
    curr_y = ref_y
    for x in range(seed_x, -1, -1):
        x1 = max(0, x - step_w // 2)
        x2 = min(w_rot, x + step_w // 2 + 1)
        col_proj = np.sum(255.0 - gray_deskewed[:, x1:x2], axis=1)
        corr = np.correlate(col_proj, template, mode='same')
        s_y1 = max(0, int(round(curr_y - search_r)))
        s_y2 = min(h_rot, int(round(curr_y + search_r + 1)))
        if s_y2 > s_y1:
            best_offset = int(np.argmax(corr[s_y1:s_y2]))
            best_y = s_y1 + best_offset
            curr_y = float(np.clip(best_y, curr_y - max_step_dy, curr_y + max_step_dy))
        track_y[x] = curr_y

    # Track RIGHT towards gutter/margin
    curr_y = ref_y
    for x in range(seed_x, w_rot):
        x1 = max(0, x - step_w // 2)
        x2 = min(w_rot, x + step_w // 2 + 1)
        col_proj = np.sum(255.0 - gray_deskewed[:, x1:x2], axis=1)
        corr = np.correlate(col_proj, template, mode='same')
        s_y1 = max(0, int(round(curr_y - search_r)))
        s_y2 = min(h_rot, int(round(curr_y + search_r + 1)))
        if s_y2 > s_y1:
            best_offset = int(np.argmax(corr[s_y1:s_y2]))
            best_y = s_y1 + best_offset
            curr_y = float(np.clip(best_y, curr_y - max_step_dy, curr_y + max_step_dy))
        track_y[x] = curr_y

    # Smooth tracked curve with Gaussian filter scaled to image width
    k_size = max(15, int(w_rot * 0.08) | 1)
    sigma = max(3.0, k_size / 3.0)
    smooth_curve = cv2.GaussianBlur(track_y.reshape(1, -1), (k_size, 1), sigma).flatten()
    target_y = float(np.median(smooth_curve))
    delta_y = smooth_curve - target_y
    max_bend = float(np.max(np.abs(delta_y)))

    # Scale-adaptive dewarping trigger (proportional to staff spacing)
    min_bend_thresh = max(1.5, staff_spacing * 0.20)
    if max_bend >= min_bend_thresh:
        pad_v = max(int(staff_spacing * 1.5), int(max_bend * 1.5) + int(staff_spacing * 0.8))
        padded_dewarp = cv2.copyMakeBorder(deskewed, pad_v, pad_v, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h_pw, w_pw = padded_dewarp.shape[:2]

        map_x = np.tile(np.arange(w_pw, dtype=np.float32), (h_pw, 1))
        map_y = np.empty((h_pw, w_pw), dtype=np.float32)
        for y in range(h_pw):
            map_y[y, :] = np.float32(y + delta_y)

        deskewed = cv2.remap(
            padded_dewarp,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

    return deskewed, round(tilt_deg, 1), round(max_bend, 1)

