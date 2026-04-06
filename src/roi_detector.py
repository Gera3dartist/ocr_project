"""Counter window detection and black/red region separation."""

import cv2
import numpy as np

from src.config import Config


def detect_window_contour(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Dynamically detect the counter window rectangle.

    Args:
        gray: Preprocessed grayscale image.

    Returns:
        (x, y, w, h) bounding rectangle of the counter window.

    Raises:
        ValueError: If no suitable rectangle is found.
    """
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ratio = w / h
        area = w * h
        if 2.0 < ratio < 8.0 and area > best_area and area > 2000:
            best = (x, y, w, h)
            best_area = area

    if best is None:
        raise ValueError("Cannot detect counter window in image")
    return best


def separate_black_red(color_roi: np.ndarray) -> int:
    """Find x-coordinate where the red background begins.

    Args:
        color_roi: BGR color image of the counter window.

    Returns:
        Column index where red section starts.
    """
    hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2

    h = color_roi.shape[0]
    col_sums = np.sum(red_mask > 0, axis=0)
    red_cols = np.where(col_sums > h * 0.1)[0]

    if len(red_cols) > 0:
        return int(red_cols[0])

    # Fallback: assume 5/8 ratio
    return int(color_roi.shape[1] * 0.625)


def _detect_tilt_angle(region: np.ndarray) -> float:
    """Detect the tilt angle of the dark counter band.

    Finds the top and bottom edges of the dark band by scanning each column
    for the dark/light transition, then fits a line to those edge points.

    Args:
        region: Grayscale image of the counter region.

    Returns:
        Tilt angle in degrees. Positive = counter-clockwise.
    """
    median_val = float(np.median(region))
    edges = []
    for col in range(region.shape[1]):
        dark_rows = np.where(region[:, col] < median_val)[0]
        if len(dark_rows) > 2:
            edges.append((col, dark_rows[0]))
            edges.append((col, dark_rows[-1]))

    if len(edges) < 20:
        return 0.0

    pts = np.array(edges, dtype=np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    return float(np.degrees(np.arctan2(vy[0], vx[0])))


def deskew(region: np.ndarray) -> np.ndarray:
    """Remove rotation from a grayscale region to compensate for camera tilt.

    Detects the tilt angle by finding the dark/light boundary edges of the
    counter band, then rotates to straighten.

    Args:
        region: Grayscale image of the counter region (dark band with digits).

    Returns:
        Deskewed grayscale image, cropped to the dark counter content.
    """
    angle = _detect_tilt_angle(region)

    if abs(angle) < 0.3:
        return region

    h, w = region.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding size to avoid clipping
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(region, M, (new_w, new_h), borderValue=int(np.median(region)))

    # Crop to the dark band in the rotated result
    median_val = float(np.median(rotated))
    _, dark_mask = cv2.threshold(rotated, int(median_val), 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    coords = cv2.findNonZero(dark_mask)
    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        rotated = rotated[y : y + ch, x : x + cw]

    return rotated



def find_counter_window(
    gray: np.ndarray, color: np.ndarray, config: Config
) -> np.ndarray:
    """Extract the black-background digit region from the meter image.

    Uses calibrated ROI from config, with dynamic contour detection as fallback.

    Args:
        gray: Preprocessed grayscale image.
        color: Resized BGR color image (same dimensions as gray).
        config: Pipeline configuration with ROI coordinates.

    Returns:
        Grayscale image of the black-background digit region (5 digits).
    """
    h, w = gray.shape
    roi = config.roi

    # Use calibrated ROI with 10% margin expansion
    x = int(roi.x_norm * w)
    y = int(roi.y_norm * h)
    rw = int(roi.w_norm * w)
    rh = int(roi.h_norm * h)

    margin_x = int(rw * 0.1)
    margin_y = int(rh * 0.1)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    rw = min(w - x, rw + 2 * margin_x)
    rh = min(h - y, rh + 2 * margin_y)

    # Try dynamic detection within the ROI region
    gray_crop = gray[y : y + rh, x : x + rw]
    color_crop = color[y : y + rh, x : x + rw]

    try:
        dx, dy, dw, dh = detect_window_contour(gray_crop)
        window_gray = gray_crop[dy : dy + dh, dx : dx + dw]
        window_color = color_crop[dy : dy + dh, dx : dx + dw]
    except ValueError:
        # Fallback: use the calibrated ROI directly
        window_gray = gray_crop
        window_color = color_crop

    # Separate black from red section
    boundary = separate_black_red(window_color)
    return window_gray[:, :boundary]
