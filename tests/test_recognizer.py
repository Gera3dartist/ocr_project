from pathlib import Path

import cv2
import numpy as np
import pytest

from src.recognizer import (
    load_templates,
    recognize_digit,
    recognize_all,
    detect_transition,
)

TEMPLATES_PATH = str(Path(__file__).parent.parent / "templates" / "templates.npz")


@pytest.fixture
def templates() -> dict[int, np.ndarray]:
    return load_templates(TEMPLATES_PATH)


def test_load_templates_has_all_digits(templates: dict[int, np.ndarray]) -> None:
    assert set(templates.keys()) == set(range(10))


def test_load_templates_correct_shape(templates: dict[int, np.ndarray]) -> None:
    for digit, tmpl in templates.items():
        assert tmpl.shape == (60, 40), f"Digit {digit} shape: {tmpl.shape}"
        assert tmpl.dtype == np.uint8


def test_recognize_real_template(templates: dict[int, np.ndarray]) -> None:
    """A template should match itself with high confidence."""
    for digit in [0, 1, 3, 4, 8]:  # real templates from CANDIDATE.jpg
        result, confidence = recognize_digit(templates[digit], templates)
        assert result == digit, f"Expected {digit}, got {result} (conf={confidence:.3f})"
        assert confidence > 0.3


def test_recognize_synthetic_template(templates: dict[int, np.ndarray]) -> None:
    """Synthetic templates should match themselves."""
    for digit in [2, 5, 6, 7, 9]:
        result, confidence = recognize_digit(templates[digit], templates)
        assert result == digit, f"Expected {digit}, got {result}"


def test_recognize_all_returns_meter_reading(templates: dict[int, np.ndarray]) -> None:
    digit_images = [templates[d] for d in [0, 3, 8, 1, 4]]
    reading = recognize_all(digit_images, templates)
    assert reading.digits == "03814"
    assert len(reading.confidence) == 5
    assert len(reading.transitioning) == 5


def test_detect_transition_on_normal_digit(templates: dict[int, np.ndarray]) -> None:
    """A clean digit template should not be flagged as transitioning."""
    is_trans = detect_transition(templates[3], templates)
    assert is_trans is False


def test_detect_transition_on_shifted_digit(templates: dict[int, np.ndarray]) -> None:
    """A vertically shifted digit should be flagged as transitioning."""
    # Create a synthetic transitioning digit: shift digit 3 down by 40%
    tmpl_3 = templates[3]
    tmpl_4 = templates[4]
    h = tmpl_3.shape[0]
    shift = int(h * 0.4)

    # Composite: top part from digit 4, bottom part from digit 3
    composite = np.zeros_like(tmpl_3)
    composite[:shift, :] = tmpl_4[h - shift :, :]
    composite[shift:, :] = tmpl_3[: h - shift, :]

    is_trans = detect_transition(composite, templates)
    assert is_trans is True
