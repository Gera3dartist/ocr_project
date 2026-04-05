"""Build digit templates from meter images.

Usage:
    python -m tools.build_templates images/CANDIDATE.jpg --labels 0,3,8,1,4
    python -m tools.build_templates --synthetic-only

The first form extracts digits from an image and saves them as templates.
The second form generates synthetic templates for all digits 0-9.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, ROI, load_config
from src.preprocessing import load_and_prepare
from src.roi_detector import find_counter_window
from src.segmenter import segment_digits

TEMPLATE_SIZE = (40, 60)
PROJECT_ROOT = Path(__file__).parent.parent


def extract_from_image(
    image_path: str,
    labels: list[int],
    config: Config,
) -> dict[int, np.ndarray]:
    """Extract digit templates from a meter image.

    Args:
        image_path: Path to meter image.
        labels: Digit label for each of the 5 positions.
        config: Pipeline config.

    Returns:
        Dict mapping digit value to template image.
    """
    gray, color = load_and_prepare(image_path, config.working_width)
    black_region = find_counter_window(gray, color, config)
    digit_images = segment_digits(
        black_region, config.num_digits, config.template_size
    )

    templates: dict[int, np.ndarray] = {}
    for label, img in zip(labels, digit_images):
        if label not in templates:
            templates[label] = img
    return templates


def generate_synthetic_templates(
    template_size: tuple[int, int] = TEMPLATE_SIZE,
) -> dict[int, np.ndarray]:
    """Generate synthetic digit templates using OpenCV text rendering.

    These approximate mechanical counter digits. Real templates from
    actual meter images will always be more accurate.

    Args:
        template_size: (width, height) of each template.

    Returns:
        Dict mapping digit 0-9 to template image.
    """
    tw, th = template_size
    templates: dict[int, np.ndarray] = {}

    for digit in range(10):
        img = np.zeros((th, tw), dtype=np.uint8)
        text = str(digit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 2

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (tw - text_size[0]) // 2
        y = (th + text_size[1]) // 2

        cv2.putText(img, text, (x, y), font, scale, 255, thickness)
        templates[digit] = img

    return templates


def save_templates(templates: dict[int, np.ndarray], output_path: str) -> None:
    """Save templates to .npz archive.

    Args:
        templates: Dict mapping digit value to template image.
        output_path: Path for the .npz file.
    """
    arrays = {str(k): v for k, v in sorted(templates.items())}
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build digit templates")
    parser.add_argument("image", nargs="?", help="Path to meter image")
    parser.add_argument(
        "--labels",
        help="Comma-separated digit labels for 5 positions (e.g., 0,3,8,1,4)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Generate only synthetic templates",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "templates" / "templates.npz"),
        help="Output path for templates.npz",
    )
    args = parser.parse_args()

    # Start with synthetic templates for all digits
    templates = generate_synthetic_templates()
    print(f"Generated synthetic templates for digits 0-9")

    # Override with real templates if image provided
    if args.image and args.labels:
        labels = [int(x) for x in args.labels.split(",")]
        if len(labels) != 5:
            parser.error("--labels must have exactly 5 comma-separated digits")

        config = load_config(args.config)
        real_templates = extract_from_image(args.image, labels, config)
        templates.update(real_templates)
        print(f"Extracted real templates for digits: {sorted(real_templates.keys())}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_templates(templates, args.output)
    print(f"Saved {len(templates)} templates to {args.output}")


if __name__ == "__main__":
    main()
