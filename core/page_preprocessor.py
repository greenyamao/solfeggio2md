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


def normalize_staff_crop(crop_bgr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    High-precision music staff crop normalization:
    1. Determines precise rotational tilt angle using 2D-DFT (2D Discrete Fourier Transform).
    2. Performs high-quality rotational deskew with border padding so no notes are clipped.
    3. Analyzes genuine 5-line staff straightness across vertical slices (immune to notes/beams).
    4. Applies subpixel dewarping ONLY when true multi-line physical spine arching (>= 3.5 px) is present.
    
    Returns:
        (normalized_crop_bgr, tilt_angle_degrees, bend_delta_pixels)
    """
    h_orig, w_orig = crop_bgr.shape[:2]
    if h_orig < 20 or w_orig < 50:
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

    # 2. Rotate to exact horizontal orientation
    angle_rad = abs(np.radians(tilt_deg))
    pad_rot = int(np.ceil(w_orig * np.sin(angle_rad) / 2.0)) + 2 if abs(tilt_deg) >= 0.1 else 0

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

    # 3. Detect 5-line staff line structures across slices to measure curvature
    gray_deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_deskewed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_len = max(20, w_rot // 25)
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    lines_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_k)

    num_slices = min(15, max(7, w_rot // 45))
    slice_xs = np.linspace(w_rot * 0.10, w_rot * 0.90, num_slices)
    col_hw = max(4, int(w_rot * 0.02))

    sample_x = []
    sample_y = []

    for sx in slice_xs:
        x1 = max(0, int(sx - col_hw))
        x2 = min(w_rot, int(sx + col_hw))
        sl = lines_only[:, x1:x2]
        proj = np.sum(sl, axis=1)

        peaks = []
        for y in range(1, h_rot - 1):
            if proj[y] > (x2 - x1) * 255 * 0.35 and proj[y] >= proj[y-1] and proj[y] >= proj[y+1]:
                peaks.append(y)

        filtered = []
        for p in peaks:
            if not filtered or p - filtered[-1] > 3:
                filtered.append(p)
            elif proj[p] > proj[filtered[-1]]:
                filtered[-1] = p

        # 5-line staff pattern: 5 equidistant peaks with spacing S in [5, 25] px
        if len(filtered) >= 5:
            for i in range(len(filtered) - 4):
                grp = filtered[i:i+5]
                diffs = np.diff(grp)
                mean_s = np.mean(diffs)
                if 5 <= mean_s <= 25 and np.max(np.abs(diffs - mean_s)) <= 3.0:
                    sample_x.append(float(sx))
                    sample_y.append(float(np.mean(grp)))

    bend_delta = 0.0
    if len(sample_x) >= 6:
        sx_arr = np.array(sample_x)
        sy_arr = np.array(sample_y)

        # Separate grand staff staves cleanly by image vertical midpoint
        mid_y = h_rot / 2.0
        upper_mask = sy_arr < mid_y
        lower_mask = sy_arr >= mid_y

        if np.sum(upper_mask) >= 5:
            fit_x, fit_y = sx_arr[upper_mask], sy_arr[upper_mask]
        elif np.sum(lower_mask) >= 5:
            fit_x, fit_y = sx_arr[lower_mask], sy_arr[lower_mask]
        else:
            fit_x, fit_y = sx_arr, sy_arr

        if len(fit_x) >= 5:
            poly = np.polyfit(fit_x, fit_y, 2)
            xs_all = np.arange(w_rot)
            curve_y = np.polyval(poly, xs_all)
            chord = np.linspace(curve_y[0], curve_y[-1], w_rot)
            bend_delta = float(np.max(np.abs(curve_y - chord)))

            # Only dewarp if genuine physical curvature is >= 3.5 px
            if bend_delta >= 3.5:
                pad_warp = max(12, int(bend_delta * 1.5))
                padded_warp = cv2.copyMakeBorder(deskewed, pad_warp, pad_warp, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                h_pw, w_pw = padded_warp.shape[:2]

                poly_pw = np.polyfit(fit_x, fit_y + pad_warp, 2)
                curve_pw = np.polyval(poly_pw, np.arange(w_pw))
                target_y = float(np.mean(curve_pw))
                shift_y = curve_pw - target_y

                map_x = np.tile(np.arange(w_pw, dtype=np.float32), (h_pw, 1))
                map_y = np.empty((h_pw, w_pw), dtype=np.float32)
                for y in range(h_pw):
                    map_y[y, :] = np.float32(y + shift_y)

                deskewed = cv2.remap(
                    padded_warp,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )

    return deskewed, round(tilt_deg, 1), round(bend_delta, 1)
