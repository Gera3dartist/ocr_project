from pathlib import Path

import numpy as np

from src.config import Config, ROI
from src.preprocessing import load_and_prepare
from src.roi_detector import find_counter_window
from src.segmenter import binarize_region, segment_digits

CANDIDATE = str(Path(__file__).parent.parent / "images" / "CANDIDATE_ready.jpg")


def _get_black_region() -> np.ndarray:
    gray, color = load_and_prepare(CANDIDATE, working_width=640)
    config = Config(roi=ROI(x_norm=0.44, y_norm=0.44, w_norm=0.45, h_norm=0.14))
    return find_counter_window(gray, color, config)


def test_segment_returns_five_digits() -> None:
    region = _get_black_region()
    digits = segment_digits(region, num_digits=5, template_size=(40, 60))
    assert len(digits) == 5


def test_segment_correct_dimensions() -> None:
    region = _get_black_region()
    digits = segment_digits(region, num_digits=5, template_size=(40, 60))
    for d in digits:
        assert d.shape == (60, 40), f"Expected (60, 40), got {d.shape}"


def test_segment_dtype_uint8() -> None:
    region = _get_black_region()
    digits = segment_digits(region, num_digits=5, template_size=(40, 60))
    for d in digits:
        assert d.dtype == np.uint8


def test_segment_is_binary() -> None:
    region = _get_black_region()
    digits = segment_digits(region, num_digits=5, template_size=(40, 60))
    for d in digits:
        unique = set(np.unique(d))
        assert unique.issubset({0, 255}), f"Non-binary values: {unique}"


def test_segment_has_white_pixels() -> None:
    """Each digit should have some white pixels (the digit itself)."""
    region = _get_black_region()
    digits = segment_digits(region, num_digits=5, template_size=(40, 60))
    for i, d in enumerate(digits):
        white_ratio = np.sum(d == 255) / d.size
        assert white_ratio > 0.02, f"Digit {i} has no white pixels ({white_ratio:.3f})"


# --- binarize_region tests ---


def test_binarize_dim_image_uses_raw_otsu() -> None:
    """Dim images (OTSU <= 120) should use the raw OTSU threshold."""
    # Dark background (30) with bright digits (120) → OTSU ~75
    region = np.full((20, 100), 30, dtype=np.uint8)
    region[5:15, 10:20] = 120  # a bright block
    region[5:15, 50:60] = 120

    binary = binarize_region(region)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
    # The bright blocks should be white
    assert np.sum(binary[5:15, 10:20] == 255) > 50


def test_binarize_bright_image_lowers_threshold() -> None:
    """Bright images (OTSU > 120) should lower threshold to keep thin strokes.

    Under strong LED lighting, OTSU overshoots and erodes thin digit
    strokes. The adaptive reduction preserves them.
    """
    # Bright background (100) with very bright digits (220) → OTSU ~160
    region = np.full((30, 100), 100, dtype=np.uint8)
    # Thick digit stroke — survives even high threshold
    region[5:25, 20:30] = 220
    # Thin stroke at 140 — lost by raw OTSU (~160) but kept by reduction
    region[5:25, 60:63] = 140

    binary_reduced = binarize_region(region)
    # The thin stroke column should have SOME white pixels
    thin_white = np.sum(binary_reduced[5:25, 60:63] == 255)
    assert thin_white > 0, "Thin stroke lost — reduction not working"


def test_binarize_region_returns_binary() -> None:
    region = np.random.randint(0, 256, (20, 80), dtype=np.uint8)
    binary = binarize_region(region)
    assert set(np.unique(binary)).issubset({0, 255})
