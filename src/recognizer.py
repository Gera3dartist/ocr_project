"""Digit recognition via template matching with multi-variant support."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MeterReading:
    """Result of a meter reading attempt."""

    digits: str
    confidence: list[float]
    transitioning: list[bool]


def load_templates(path: str) -> dict[int, list[np.ndarray]]:
    """Load digit templates from .npz archive.

    Supports multiple variants per digit (keys like "0", "0_v1", "0_v2").
    More variants from different camera angles improve recognition robustness.

    Args:
        path: Path to templates.npz file.

    Returns:
        Dict mapping digit value (0-9) to list of template images.
    """
    data = np.load(path)
    templates: dict[int, list[np.ndarray]] = {}
    for k in data.files:
        digit = int(k.split("_")[0])
        if digit not in templates:
            templates[digit] = []
        templates[digit].append(data[k])
    return templates


def _score_template(digit_img: np.ndarray, tmpl: np.ndarray) -> float:
    """Score a digit image against a single template.

    Uses hybrid scoring: normalized cross-correlation weighted with pixel
    overlap (IoU). The combination resists both noise (correlation is
    robust) and shape differences (IoU catches silhouette mismatches).

    Args:
        digit_img: Binarized digit image.
        tmpl: Template image (same size or will be resized).

    Returns:
        Combined score in roughly [0, 1] range.
    """
    if digit_img.shape != tmpl.shape:
        img = cv2.resize(digit_img, (tmpl.shape[1], tmpl.shape[0]))
    else:
        img = digit_img

    result = cv2.matchTemplate(
        img.astype(np.float32),
        tmpl.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    correlation = float(result[0, 0])

    img_white = img > 127
    tmpl_white = tmpl > 127
    intersection = np.sum(img_white & tmpl_white)
    union = np.sum(img_white | tmpl_white)
    overlap = float(intersection / union) if union > 0 else 0.0

    return 0.6 * correlation + 0.4 * overlap


def recognize_digit(
    digit_img: np.ndarray, templates: dict[int, list[np.ndarray]]
) -> tuple[int, float]:
    """Recognize a single digit by matching against all template variants.

    Confidence is the gap between the best and second-best score: a large
    gap means the winner is unambiguous.

    Args:
        digit_img: Binarized digit image (same size as templates).
        templates: Dict mapping digit value to list of template images.

    Returns:
        Tuple of (recognized digit, confidence score).
    """
    scores: list[tuple[int, float]] = []

    for digit, tmpls in templates.items():
        best = max(_score_template(digit_img, t) for t in tmpls)
        scores.append((digit, best))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_digit, best_score = scores[0]
    second_score = scores[1][1]
    confidence = best_score - second_score

    return best_digit, confidence


def recognize_all(
    digit_images: list[np.ndarray], templates: dict[int, list[np.ndarray]]
) -> MeterReading:
    """Recognize all digits and return a complete meter reading.

    Args:
        digit_images: List of binarized digit images.
        templates: Digit templates with multiple variants per digit.

    Returns:
        MeterReading with digits, confidence, and transition flags.
    """
    digits_str = ""
    confidences: list[float] = []

    for img in digit_images:
        digit, confidence = recognize_digit(img, templates)
        digits_str += str(digit)
        confidences.append(round(confidence, 3))

    return MeterReading(
        digits=digits_str,
        confidence=confidences,
        transitioning=[False] * len(digit_images),
    )
