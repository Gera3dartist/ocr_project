"""Build digit templates from meter images.

Usage:
    # Extract from multiple images (recommended for robustness):
    python -m tools.build_templates \\
        images/img1.jpg:0,3,8,3,3 \\
        images/img2.jpg:0,3,8,3,3

    # Single image:
    python -m tools.build_templates images/CANDIDATE.jpg --labels 0,3,8,1,4

    # Synthetic only (fallback for digits with no real samples):
    python -m tools.build_templates --synthetic-only

Multiple images from different camera angles build a variant library that
makes recognition robust to small changes in position, tilt, and lighting.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import Config, load_config
from src.preprocessing import load_and_prepare
from src.roi_detector import find_counter_window
from src.segmenter import segment_digits

TEMPLATE_SIZE = (40, 60)
PROJECT_ROOT = Path(__file__).parent.parent


def extract_from_image(
    image_path: str,
    labels: list[int],
    config: Config,
) -> list[tuple[int, np.ndarray]]:
    """Extract digit templates from a meter image.

    Args:
        image_path: Path to meter image.
        labels: Digit label for each of the 5 positions.
        config: Pipeline config.

    Returns:
        List of (digit_label, template_image) pairs.
    """
    gray, color = load_and_prepare(image_path, config.working_width)
    black_region = find_counter_window(gray, color, config)
    digit_images = segment_digits(
        black_region, config.num_digits, config.template_size
    )
    return list(zip(labels, digit_images))


def generate_synthetic_templates(
    template_size: tuple[int, int] = TEMPLATE_SIZE,
) -> dict[int, np.ndarray]:
    """Generate synthetic digit templates using OpenCV text rendering.

    Uses thick, bold rendering to approximate mechanical counter digits.
    Real templates from actual meter images will always be more accurate.

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
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.4
        thickness = 3

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (tw - text_size[0]) // 2
        y = (th + text_size[1]) // 2

        cv2.putText(img, text, (x, y), font, scale, 255, thickness)

        # Dilate slightly to approximate the thick mechanical counter style
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.dilate(img, kern, iterations=1)

        templates[digit] = img

    return templates


def save_templates(
    templates: dict[str, np.ndarray], output_path: str
) -> None:
    """Save templates to .npz archive.

    Args:
        templates: Dict mapping template key ("0", "0_v1", ...) to image.
        output_path: Path for the .npz file.
    """
    np.savez_compressed(output_path, **templates)


def build_template_dict(
    pairs: list[tuple[int, np.ndarray]],
    synthetic: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Merge real extracted templates with synthetic fallbacks.

    Real templates are stored as variants: "3", "3_v1", "3_v2", etc.
    Synthetic templates fill in digits that have no real samples.

    Args:
        pairs: List of (digit_label, image) from extract_from_image calls.
        synthetic: Synthetic templates for digits 0-9.

    Returns:
        Dict mapping template key to template image, ready for save.
    """
    # Count variants per digit
    variant_count: dict[int, int] = {}
    result: dict[str, np.ndarray] = {}

    for label, img in pairs:
        count = variant_count.get(label, 0)
        key = str(label) if count == 0 else f"{label}_v{count}"
        result[key] = img
        variant_count[label] = count + 1

    # Fill missing digits with synthetic templates
    for digit in range(10):
        if digit not in variant_count:
            result[str(digit)] = synthetic[digit]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build digit templates from meter images"
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="image_path:labels pairs (e.g. img.jpg:0,3,8,3,3) "
        "or just image path when --labels is used",
    )
    parser.add_argument(
        "--labels",
        help="Comma-separated digit labels (for single-image mode)",
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

    synthetic = generate_synthetic_templates()
    print("Generated synthetic templates for digits 0-9")

    # Parse source images
    image_label_pairs: list[tuple[str, list[int]]] = []

    if args.sources:
        for source in args.sources:
            if ":" in source:
                # Format: image_path:0,3,8,3,3
                path, label_str = source.rsplit(":", 1)
                labels = [int(x) for x in label_str.split(",")]
            elif args.labels:
                # Single image with --labels flag
                path = source
                labels = [int(x) for x in args.labels.split(",")]
            else:
                parser.error(
                    "Use image:labels format or provide --labels"
                )
            if len(labels) != 5:
                parser.error(f"Need exactly 5 labels, got {len(labels)} for {path}")
            image_label_pairs.append((path, labels))

    # Extract templates from all source images
    all_pairs: list[tuple[int, np.ndarray]] = []
    if image_label_pairs:
        config = load_config(args.config)
        for path, labels in image_label_pairs:
            extracted = extract_from_image(path, labels, config)
            all_pairs.extend(extracted)
            print(f"Extracted from {path}: digits {labels}")

    templates = build_template_dict(all_pairs, synthetic)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_templates(templates, args.output)

    # Summary
    real_digits = set()
    variant_counts: dict[int, int] = {}
    for key in templates:
        d = int(key.split("_")[0])
        variant_counts[d] = variant_counts.get(d, 0) + 1
        if key not in {str(i) for i in range(10)} or all_pairs:
            real_digits.add(d)

    print(f"Saved {len(templates)} templates to {args.output}")
    for d in sorted(variant_counts):
        source = "real" if d in {label for label, _ in all_pairs} else "synthetic"
        print(f"  Digit {d}: {variant_counts[d]} variant(s) ({source})")


if __name__ == "__main__":
    main()
