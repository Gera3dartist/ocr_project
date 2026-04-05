from pathlib import Path

from src.pipeline import read_meter
from src.recognizer import MeterReading

PROJECT_ROOT = Path(__file__).parent.parent
CANDIDATE = str(PROJECT_ROOT / "images" / "CANDIDATE.jpg")
CONFIG = str(PROJECT_ROOT / "config.json")
TEMPLATES = str(PROJECT_ROOT / "templates" / "templates.npz")


def test_read_meter_returns_reading() -> None:
    reading = read_meter(CANDIDATE, CONFIG, TEMPLATES)
    assert isinstance(reading, MeterReading)
    assert len(reading.digits) == 5
    assert len(reading.confidence) == 5
    assert len(reading.transitioning) == 5


def test_read_meter_all_digits_are_numeric() -> None:
    reading = read_meter(CANDIDATE, CONFIG, TEMPLATES)
    assert reading.digits.isdigit()


def test_read_meter_correct_reading() -> None:
    """The CANDIDATE.jpg meter reads 03814 on the black section."""
    reading = read_meter(CANDIDATE, CONFIG, TEMPLATES)
    # Allow tolerance: at least 4 of 5 digits correct
    expected = "03814"
    matches = sum(a == b for a, b in zip(reading.digits, expected))
    assert matches >= 4, (
        f"Expected ~'{expected}', got '{reading.digits}' ({matches}/5 correct)"
    )


def test_read_meter_confidence_positive() -> None:
    reading = read_meter(CANDIDATE, CONFIG, TEMPLATES)
    for i, conf in enumerate(reading.confidence):
        assert conf >= 0.0, f"Digit {i} has negative confidence: {conf}"
