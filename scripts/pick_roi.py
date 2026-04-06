#!/usr/bin/env python3
"""Interactively pick ROI on a captured image and save to config.json.

Usage:
    python scripts/pick_roi.py <image_path> [--config config.json]

Controls:
    - Click and drag to select the ROI rectangle
    - ENTER/SPACE to confirm selection
    - 'c' to cancel and re-select
    - ESC to quit without saving
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def pick_roi(image_path: str) -> tuple[float, float, float, float] | None:
    """Display image and let user draw an ROI rectangle.

    Returns:
        Normalized (x, y, w, h) or None if cancelled.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image '{image_path}'")
        sys.exit(1)

    h, w = image.shape[:2]
    window_name = "Select ROI - ENTER to confirm, ESC to cancel"

    roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        return None

    return (rx / w, ry / h, rw / w, rh / h)


def save_roi(
    config_path: str, x_norm: float, y_norm: float, w_norm: float, h_norm: float
) -> None:
    """Update the roi section in config.json."""
    path = Path(config_path)
    config = json.loads(path.read_text()) if path.exists() else {}

    config["roi"] = {
        "x_norm": round(x_norm, 4),
        "y_norm": round(y_norm, 4),
        "w_norm": round(w_norm, 4),
        "h_norm": round(h_norm, 4),
    }

    path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick ROI and save to config.json")
    parser.add_argument("image", help="Path to meter image")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config.json"),
        help="Path to config.json (default: project root config.json)",
    )
    args = parser.parse_args()

    result = pick_roi(args.image)
    if result is None:
        print("Selection cancelled.")
        sys.exit(0)

    x_norm, y_norm, w_norm, h_norm = result
    print(f"Selected ROI (normalized): x={x_norm:.4f} y={y_norm:.4f} w={w_norm:.4f} h={h_norm:.4f}")

    save_roi(args.config, x_norm, y_norm, w_norm, h_norm)
    print(f"Saved to {args.config}")


if __name__ == "__main__":
    main()
