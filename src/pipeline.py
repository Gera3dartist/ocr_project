"""Main OCR pipeline for SAMGAZ G4 gas meter."""

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.preprocessing import load_and_prepare
from src.recognizer import MeterReading, load_templates, recognize_all
from src.roi_detector import find_counter_window
from src.segmenter import segment_digits

PROJECT_ROOT = Path(__file__).parent.parent


def read_meter(
    image_path: str,
    config_path: str | None = None,
    templates_path: str | None = None,
) -> MeterReading:
    """Read the first 5 digits from a SAMGAZ G4 gas meter image.

    Args:
        image_path: Path to the meter image.
        config_path: Path to config.json. Defaults to project config.
        templates_path: Path to templates.npz. Defaults to project templates.

    Returns:
        MeterReading with digit string, confidence, and transition flags.
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config.json")
    if templates_path is None:
        templates_path = str(PROJECT_ROOT / "templates" / "templates.npz")

    config = load_config(config_path)
    templates = load_templates(templates_path)

    gray, color = load_and_prepare(image_path, config.working_width)
    black_region = find_counter_window(gray, color, config)
    digit_images = segment_digits(
        black_region, config.num_digits, config.template_size
    )
    reading = recognize_all(digit_images, templates)

    return reading


def main() -> None:
    parser = argparse.ArgumentParser(description="Read SAMGAZ G4 gas meter")
    parser.add_argument("image", help="Path to meter image")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--templates", default=None, help="Path to templates.npz")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    reading = read_meter(args.image, args.config, args.templates)

    if args.json:
        output = {
            "digits": reading.digits,
            "confidence": reading.confidence,
            "transitioning": reading.transitioning,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Reading: {reading.digits}")
        for i, (d, c, t) in enumerate(
            zip(reading.digits, reading.confidence, reading.transitioning)
        ):
            status = " (transitioning)" if t else ""
            print(f"  Digit {i}: {d}  confidence={c:.3f}{status}")


if __name__ == "__main__":
    main()
