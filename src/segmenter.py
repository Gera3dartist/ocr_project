"""Digit segmentation via fixed-width slicing."""

import cv2
import numpy as np


def segment_digits(
    black_region: np.ndarray,
    num_digits: int = 5,
    template_size: tuple[int, int] = (40, 60),
) -> list[np.ndarray]:
    """Segment the black-background region into individual digit images.

    Uses fixed-width slicing since mechanical counter digits are uniformly spaced.

    Args:
        black_region: Grayscale image of the black-background digit area.
        num_digits: Number of digits to extract.
        template_size: (width, height) to resize each digit to.

    Returns:
        List of binarized digit images, each sized to template_size.
    """
    # OTSU works better than adaptive on these small counter regions where the
    # digit/background contrast is bimodal. Morph open removes speckle noise.
    _, binary_full = cv2.threshold(
        black_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, kernel)

    h, w = black_region.shape
    slot_width = w / num_digits
    margin = int(slot_width * 0.08)
    tw, th = template_size

    digits = []
    for i in range(num_digits):
        x_start = int(i * slot_width) + margin
        x_end = int((i + 1) * slot_width) - margin
        binary = binary_full[:, x_start:x_end]

        # Crop vertically to the digit content
        binary = _crop_to_content(binary)

        # Resize to template size
        digit = cv2.resize(binary, (tw, th), interpolation=cv2.INTER_AREA)

        # Re-threshold after resize to keep binary
        _, digit = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY)

        digits.append(digit)

    return digits


def _crop_to_content(binary: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of white pixels, with padding.

    Args:
        binary: Binary image (0 and 255).

    Returns:
        Cropped binary image. Returns original if no white pixels found.
    """
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary

    x, y, w, h = cv2.boundingRect(coords)
    pad_y = max(2, int(h * 0.05))
    pad_x = max(2, int(w * 0.05))

    y_start = max(0, y - pad_y)
    y_end = min(binary.shape[0], y + h + pad_y)
    x_start = max(0, x - pad_x)
    x_end = min(binary.shape[1], x + w + pad_x)

    return binary[y_start:y_end, x_start:x_end]
