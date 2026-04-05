"""Image preprocessing for gas meter OCR."""

import cv2
import numpy as np


def load_and_prepare(
    path: str, working_width: int = 640
) -> tuple[np.ndarray, np.ndarray]:
    """Load image, resize, and prepare for OCR pipeline.

    Args:
        path: Path to the input image.
        working_width: Target width in pixels. Height scales proportionally.

    Returns:
        Tuple of (grayscale, color) images, both resized and preprocessed.

    Raises:
        FileNotFoundError: If image cannot be loaded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    h, w = img.shape[:2]
    scale = working_width / w
    new_h = int(h * scale)
    color = cv2.resize(img, (working_width, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray, color
