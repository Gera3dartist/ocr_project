"""Pre-processing pipeline used by read_meter_v2.

Differs from the v1 path (find_counter_window + segment_digits):
- Calibrated ROI + Hough deskew instead of color-mask ROI detection.
- Projection-profile drum-band cropping (longest dark row-run + outer
  high-variance column envelope) instead of fixed-shape windows.
- Uniform fixed-width slicing on the deskewed strip.
- Template-friendly binarized digit crops via to_template.
"""

import cv2
import numpy as np

from src.config import Config


def select_roi(
    gray: np.ndarray, config: Config, margin_frac: float = 0.1
) -> np.ndarray:
    """Crop the calibrated ROI from gray, expanded by margin_frac on each side."""
    h, w = gray.shape
    roi = config.roi
    x = int(roi.x_norm * w)
    y = int(roi.y_norm * h)
    rw = int(roi.w_norm * w)
    rh = int(roi.h_norm * h)
    mx = int(rw * margin_frac)
    my = int(rh * margin_frac)
    x = max(0, x - mx)
    y = max(0, y - my)
    rw = min(w - x, rw + 2 * mx)
    rh = min(h - y, rh + 2 * my)
    return gray[y : y + rh, x : x + rw]


def detect_near_horizontal_lines(
    gray: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    max_angle_deg: float = 30.0,
) -> list[tuple[float, float]]:
    """Find near-horizontal Hough segments. Returns (angle_deg, length) pairs."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=10,
    )
    if lines is None:
        return []
    out: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) <= max_angle_deg:
            out.append((float(angle), float(np.hypot(dx, dy))))
    return out


def estimate_tilt_angle(lines: list[tuple[float, float]]) -> float:
    """Aggregate detected line angles into a single tilt estimate (deg)."""
    if not lines:
        return 0.0
    return lines[0][0]


def rotate_with_expanded_canvas(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate around center by angle_deg, expanding canvas to fit."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2.0) - center[0]
    M[1, 2] += (nh / 2.0) - center[1]
    return cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew(gray: np.ndarray) -> np.ndarray:
    """Detect tilt via Hough lines and rotate to upright."""
    lines = detect_near_horizontal_lines(gray)
    angle = estimate_tilt_angle(lines)
    return rotate_with_expanded_canvas(gray, angle)


def crop_digits_band(strip: np.ndarray, edge_trim: int = 2) -> np.ndarray:
    """Auto-crop strip to the drum band and digit envelope.

    Two 1D projection profiles, each aggregated differently:
      - row-mean    (axis=1): drum is ONE contiguous dark band, so we take
                              the longest contiguous run below the cutoff.
      - column-std  (axis=0): each digit is its own busy peak with calm gaps
                              between them, so we take the OUTER envelope
                              (first busy column to last busy column).

    `edge_trim` shaves a few rows off the top and bottom of the drum band to
    avoid bezel transitions producing spurious black bars after polarity
    inversion.
    """

    def longest_run(mask: np.ndarray) -> tuple[int, int]:
        padded = np.r_[False, mask, False]
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        if len(starts) == 0:
            return 0, len(mask) - 1
        return max(zip(starts, ends), key=lambda r: r[1] - r[0])

    def outer_bounds(mask: np.ndarray) -> tuple[int, int]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0, len(mask) - 1
        return int(idx[0]), int(idx[-1])

    rows = strip.mean(axis=1).astype(np.float32)
    rows = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 9), 0).ravel()
    row_mask = rows < (rows.min() + rows.max()) / 2
    y_top, y_bot = longest_run(row_mask)
    y_top = min(y_bot, y_top + edge_trim)
    y_bot = max(y_top, y_bot - edge_trim)
    drum = strip[y_top : y_bot + 1]

    cols = drum.std(axis=0).astype(np.float32)
    cols = cv2.GaussianBlur(cols.reshape(-1, 1), (1, 9), 0).ravel()
    col_mask = cols > (cols.min() + cols.max()) / 2
    x_left, x_right = outer_bounds(col_mask)
    return drum[:, x_left : x_right + 1]


def segment_uniform(
    clean_image: np.ndarray,
    num_digits: int = 5,
    side_margin_frac: float = 0.08,
) -> list[np.ndarray]:
    """Slice clean strip into N equal-width columns with a small side margin."""
    _, w = clean_image.shape
    slot_w = w / num_digits
    margin = int(slot_w * side_margin_frac)
    out: list[np.ndarray] = []
    for i in range(num_digits):
        x0 = max(0, int(i * slot_w) + margin)
        x1 = min(w, int((i + 1) * slot_w) - margin)
        out.append(clean_image[:, x0:x1])
    return out


def to_template(
    digit_crop: np.ndarray,
    size: tuple[int, int] = (40, 60),
) -> np.ndarray:
    """Resize and Otsu-binarize so the digit is white (255) on black (0)."""
    resized = cv2.resize(digit_crop, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(
        resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if (binary == 255).sum() > (binary == 0).sum():
        binary = 255 - binary
    return binary


def prepare_clean_image(
    gray: np.ndarray,
    config: Config,
    *,
    contrast_alpha: float = 2.1,
    contrast_beta: int = 10,
) -> np.ndarray:
    """Full v2 pre-processing chain: ROI → deskew → normalize → contrast → drum crop."""
    gray_crop = select_roi(gray, config)
    deskewed = deskew(gray_crop)
    norm = cv2.normalize(deskewed, None, 0, 255, cv2.NORM_MINMAX)
    boosted = cv2.convertScaleAbs(norm, alpha=contrast_alpha, beta=contrast_beta)
    return crop_digits_band(boosted)
