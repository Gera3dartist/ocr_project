"""Digit recognition via template matching."""

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


def _score_template(
    digit_img: np.ndarray, tmpl: np.ndarray
) -> float:
    """Score a digit image against a single template.

    Args:
        digit_img: Binarized digit image.
        tmpl: Template image.

    Returns:
        Combined score (correlation + IoU).
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
    """Recognize a single digit using template matching.

    Uses hybrid scoring: template correlation + pixel overlap.
    When multiple variants exist for a digit, uses the best match.

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


def detect_transition(
    digit_img: np.ndarray, templates: dict[int, list[np.ndarray]]
) -> bool:
    """Detect if a digit is transitioning (partially scrolled).

    Splits the digit vertically and checks if top/bottom match different digits.

    Args:
        digit_img: Binarized digit image.
        templates: Digit templates.

    Returns:
        True if the digit appears to be transitioning.
    """
    h = digit_img.shape[0]
    mid = h // 2

    top_half = digit_img[:mid, :]
    bottom_half = digit_img[mid:, :]

    # Match top and bottom halves against template halves
    top_scores: list[tuple[int, float]] = []
    bottom_scores: list[tuple[int, float]] = []

    for digit, tmpls in templates.items():
        best_top = 0.0
        best_bot = 0.0
        for tmpl in tmpls:
            if digit_img.shape != tmpl.shape:
                tmpl = cv2.resize(tmpl, (digit_img.shape[1], digit_img.shape[0]))
            tmpl_top = tmpl[:mid, :]
            tmpl_bottom = tmpl[mid:, :]

            top_white = top_half > 127
            tmpl_top_white = tmpl_top > 127
            top_inter = np.sum(top_white & tmpl_top_white)
            top_union = np.sum(top_white | tmpl_top_white)
            top_iou = float(top_inter / top_union) if top_union > 0 else 0.0
            best_top = max(best_top, top_iou)

            bot_white = bottom_half > 127
            tmpl_bot_white = tmpl_bottom > 127
            bot_inter = np.sum(bot_white & tmpl_bot_white)
            bot_union = np.sum(bot_white | tmpl_bot_white)
            bot_iou = float(bot_inter / bot_union) if bot_union > 0 else 0.0
            best_bot = max(best_bot, bot_iou)

        top_scores.append((digit, best_top))
        bottom_scores.append((digit, best_bot))

    top_scores.sort(key=lambda x: x[1], reverse=True)
    bottom_scores.sort(key=lambda x: x[1], reverse=True)

    top_digit = top_scores[0][0]
    bottom_digit = bottom_scores[0][0]

    # If top and bottom match different consecutive digits, it's transitioning
    if top_digit != bottom_digit:
        diff = (bottom_digit - top_digit) % 10
        if diff == 1 or diff == 9:
            return True

    # Also check vertical center of mass deviation
    white_pixels = np.where(digit_img > 127)
    if len(white_pixels[0]) > 0:
        cy = np.mean(white_pixels[0]) / h
        if abs(cy - 0.5) > 0.15:
            return True

    return False


def recognize_all(
    digit_images: list[np.ndarray], templates: dict[int, list[np.ndarray]]
) -> MeterReading:
    """Recognize all digits and return a complete meter reading.

    Args:
        digit_images: List of binarized digit images.
        templates: Digit templates.

    Returns:
        MeterReading with digits, confidence, and transition flags.
    """
    digits_str = ""
    confidences: list[float] = []
    transitions: list[bool] = []

    for img in digit_images:
        digit, confidence = recognize_digit(img, templates)
        is_transitioning = detect_transition(img, templates)
        digits_str += str(digit)
        confidences.append(round(confidence, 3))
        transitions.append(is_transitioning)

    return MeterReading(
        digits=digits_str,
        confidence=confidences,
        transitioning=transitions,
    )
