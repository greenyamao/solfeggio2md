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
