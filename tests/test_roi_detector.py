from pathlib import Path

import numpy as np

from src.config import Config, ROI
from src.preprocessing import load_and_prepare
from src.roi_detector import find_counter_window, detect_window_contour, separate_black_red

CANDIDATE = str(Path(__file__).parent.parent / "images" / "CANDIDATE.jpg")


def _get_preprocessed() -> tuple[np.ndarray, np.ndarray]:
    return load_and_prepare(CANDIDATE, working_width=640)


def test_detect_window_contour_finds_rectangle() -> None:
    gray, _ = _get_preprocessed()
    x, y, w, h = detect_window_contour(gray)
    assert w > 0 and h > 0
    ratio = w / h
    assert 2.0 < ratio < 8.0, f"Aspect ratio {ratio:.1f} outside expected range"


def test_detect_window_contour_reasonable_size() -> None:
    gray, _ = _get_preprocessed()
    x, y, w, h = detect_window_contour(gray)
    img_h, img_w = gray.shape
    # Counter window should be 20-60% of image width
    assert 0.2 < w / img_w < 0.6
    # And 5-25% of image height
    assert 0.05 < h / img_h < 0.25


def test_separate_black_red_finds_boundary() -> None:
    _, color = _get_preprocessed()
    gray, _ = _get_preprocessed()
    x, y, w, h = detect_window_contour(gray)
    color_roi = color[y : y + h, x : x + w]
    boundary = separate_black_red(color_roi)
    # Boundary should be 50-75% of the way through (5 of 8 digits = 62.5%)
    ratio = boundary / color_roi.shape[1]
    assert 0.45 < ratio < 0.80, f"Red boundary at {ratio:.2f}, expected 0.45-0.80"


def test_find_counter_window_returns_black_region() -> None:
    gray, color = _get_preprocessed()
    config = Config(roi=ROI(x_norm=0.44, y_norm=0.44, w_norm=0.45, h_norm=0.14))
    black_region = find_counter_window(gray, color, config)
    assert black_region is not None
    assert black_region.ndim == 2
    # Should be wider than tall
    h, w = black_region.shape
    assert w > h


def test_find_counter_window_is_dark() -> None:
    gray, color = _get_preprocessed()
    config = Config(roi=ROI(x_norm=0.44, y_norm=0.44, w_norm=0.45, h_norm=0.14))
    black_region = find_counter_window(gray, color, config)
    # The black background region should have relatively low mean brightness
    assert black_region.mean() < 150
