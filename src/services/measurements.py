from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import tempfile

from src.capture import capture_image
from src.pipeline import read_meter

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    digits:  str
    confidence:  list[float]
    transitioning:  list[bool]
    timestamp:  datetime


def make_readings() -> Measurement:
    max_attempts = 3
    min_confidence = 0.10

    best = None
    for attempt in range(max_attempts):
        with tempfile.NamedTemporaryFile(
            suffix=".jpg", delete=False
        ) as tmp:
            image_path = tmp.name

        capture_image(image_path)
        reading = read_meter(image_path)

        if best is None or min(reading.confidence) > min(best.confidence):
            best = reading

        if min(reading.confidence) >= min_confidence:
            break

        logger.info(
            "Low confidence (%.3f), retrying (%d/%d)",
            min(reading.confidence),
            attempt + 1,
            max_attempts,
        )

    return Measurement(
        digits=best.digits,
        confidence=best.confidence,
        transitioning=best.transitioning,
        timestamp=datetime.now(timezone.utc).isoformat()
    )