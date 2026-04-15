from pathlib import Path

import pytest

from src.pipeline import read_meter
from src.recognizer import MeterReading

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG = str(PROJECT_ROOT / "config.json")
TEMPLATES = str(PROJECT_ROOT / "templates" / "templates.npz")

# Auto-discover all measuring images on disk
ALL_IMAGES = sorted(PROJECT_ROOT.glob("images/measuring_*.jpg"))

# Images with heavy LED glare — recognition may be degraded
GLARE_NAMES = {
    "measuring_1776282379.jpg",
}

GOOD_IMAGES = [p for p in ALL_IMAGES if p.name not in GLARE_NAMES]
GLARE_IMAGES = [p for p in ALL_IMAGES if p.name in GLARE_NAMES]


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


# --- Correctness on good images ---


@pytest.mark.parametrize(
    "path", [str(p) for p in GOOD_IMAGES], ids=[p.name for p in GOOD_IMAGES]
)
def test_good_image_reads_correctly(path: str) -> None:
    """Well-lit images must read exactly 03833."""
    reading = read_meter(path, CONFIG, TEMPLATES)
    name = Path(path).name
    assert reading.digits == "03833", (
        f"{name}: expected '03833', got '{reading.digits}'"
    )


@pytest.mark.parametrize(
    "path", [str(p) for p in GOOD_IMAGES], ids=[p.name for p in GOOD_IMAGES]
)
def test_good_image_min_confidence(path: str) -> None:
    """Well-lit images should have reasonable confidence on all digits."""
    reading = read_meter(path, CONFIG, TEMPLATES)
    name = Path(path).name
    assert min(reading.confidence) > 0.2, (
        f"{name}: min confidence {min(reading.confidence):.3f} too low"
    )


# --- Glare images: degraded but not garbage ---


@pytest.mark.parametrize(
    "path", [str(p) for p in GLARE_IMAGES], ids=[p.name for p in GLARE_IMAGES]
)
def test_glare_image_at_least_four_correct(path: str) -> None:
    """Glare images should get at least 4/5 digits right."""
    reading = read_meter(path, CONFIG, TEMPLATES)
    expected = "03833"
    matches = sum(a == b for a, b in zip(reading.digits, expected))
    name = Path(path).name
    assert matches >= 4, (
        f"{name}: expected ~'{expected}', got '{reading.digits}' "
        f"({matches}/5 correct)"
    )
