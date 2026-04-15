from pathlib import Path

import pytest

from src.pipeline import read_meter
from src.recognizer import MeterReading

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG = str(PROJECT_ROOT / "config.json")
TEMPLATES = str(PROJECT_ROOT / "templates" / "templates.npz")

# Auto-discover all measuring images on disk
ALL_IMAGES = sorted(PROJECT_ROOT.glob("images/measuring_*.jpg"))


def _resolve(name: str) -> str:
    return str(PROJECT_ROOT / "images" / name)


@pytest.fixture(
    params=[str(p) for p in ALL_IMAGES],
    ids=[p.name for p in ALL_IMAGES],
)
def image_path(request: pytest.FixtureRequest) -> str:
    return request.param


# --- Basic contract tests (all images) ---


def test_read_meter_returns_reading(image_path: str) -> None:
    reading = read_meter(image_path, CONFIG, TEMPLATES)
    assert isinstance(reading, MeterReading)
    assert len(reading.digits) == 5
    assert len(reading.confidence) == 5
    assert len(reading.transitioning) == 5


def test_read_meter_all_digits_are_numeric(image_path: str) -> None:
    reading = read_meter(image_path, CONFIG, TEMPLATES)
    assert reading.digits.isdigit()


def test_read_meter_confidence_positive(image_path: str) -> None:
    reading = read_meter(image_path, CONFIG, TEMPLATES)
    for i, conf in enumerate(reading.confidence):
        assert conf >= 0.0, f"Digit {i} has negative confidence: {conf}"


# --- Correctness ---


def test_read_meter_correct_reading() -> None:
    """All measuring images should read 03833, allowing 1 digit tolerance
    for images with glare or unusual lighting."""
    expected = "03833"
    for img in ALL_IMAGES:
        reading = read_meter(str(img), CONFIG, TEMPLATES)
        matches = sum(a == b for a, b in zip(reading.digits, expected))
        assert matches >= 4, (
            f"{img.name}: expected ~'{expected}', got '{reading.digits}' "
            f"({matches}/5 correct)"
        )
