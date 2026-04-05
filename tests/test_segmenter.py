from pathlib import Path

import numpy as np

from src.config import Config, ROI
from src.preprocessing import load_and_prepare
from src.roi_detector import find_counter_window
from src.segmenter import segment_digits

CANDIDATE = str(Path(__file__).parent.parent / "images" / "CANDIDATE.jpg")


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
