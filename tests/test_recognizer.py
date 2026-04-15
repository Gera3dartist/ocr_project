from pathlib import Path

import cv2
import numpy as np
import pytest

from src.recognizer import (
    MeterReading,
    load_templates,
    recognize_digit,
    recognize_all,
    _score_template,
)

TEMPLATES_PATH = str(Path(__file__).parent.parent / "templates" / "templates.npz")


@pytest.fixture
def templates() -> dict[int, list[np.ndarray]]:
    return load_templates(TEMPLATES_PATH)


def test_load_templates_has_all_digits(templates: dict[int, list[np.ndarray]]) -> None:
    assert set(templates.keys()) == set(range(10))


def test_load_templates_returns_lists(templates: dict[int, list[np.ndarray]]) -> None:
    for digit, variants in templates.items():
        assert isinstance(variants, list)
        assert len(variants) >= 1, f"Digit {digit} has no variants"


def test_load_templates_correct_shape(templates: dict[int, list[np.ndarray]]) -> None:
    for digit, variants in templates.items():
        for v in variants:
            assert v.shape == (60, 40), f"Digit {digit} shape: {v.shape}"
            assert v.dtype == np.uint8


def test_recognize_digit_matches_own_template(
    templates: dict[int, list[np.ndarray]],
) -> None:
    """A template should match itself with the highest score."""
    for digit in range(10):
        tmpl = templates[digit][0]
        result, confidence = recognize_digit(tmpl, templates)
        assert result == digit, f"Expected {digit}, got {result} (conf={confidence:.3f})"


def test_recognize_digit_confidence_positive(
    templates: dict[int, list[np.ndarray]],
) -> None:
    tmpl = templates[3][0]
    _, confidence = recognize_digit(tmpl, templates)
    assert confidence > 0.0


def test_recognize_all_returns_meter_reading(
    templates: dict[int, list[np.ndarray]],
) -> None:
    digit_images = [templates[d][0] for d in [0, 3, 8, 3, 3]]
    reading = recognize_all(digit_images, templates)
    assert isinstance(reading, MeterReading)
    assert reading.digits == "03833"
    assert len(reading.confidence) == 5
    assert len(reading.transitioning) == 5


def test_score_template_self_match_high() -> None:
    """Scoring an image against itself should give a high score."""
    img = np.zeros((60, 40), dtype=np.uint8)
    cv2.putText(img, "3", (5, 45), cv2.FONT_HERSHEY_DUPLEX, 1.4, 255, 3)
    score = _score_template(img, img)
    assert score > 0.9


def test_score_template_mismatch_low() -> None:
    """Scoring completely different images should give a low score."""
    img = np.zeros((60, 40), dtype=np.uint8)
    img[:30, :] = 255  # top half white

    tmpl = np.zeros((60, 40), dtype=np.uint8)
    tmpl[30:, :] = 255  # bottom half white

    score = _score_template(img, tmpl)
    assert score < 0.3


def test_multi_variant_all_digits_recognized(
    templates: dict[int, list[np.ndarray]],
) -> None:
    """Every digit's first variant should be recognized as itself."""
    for digit in range(10):
        result, _ = recognize_digit(templates[digit][0], templates)
        assert result == digit, f"Digit {digit} first variant misrecognized as {result}"
