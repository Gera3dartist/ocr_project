from pathlib import Path

import numpy as np

from src.preprocessing import load_and_prepare

CANDIDATE = str(Path(__file__).parent.parent / "images" / "CANDIDATE_ready.jpg")


def test_load_returns_gray_and_color() -> None:
    gray, color = load_and_prepare(CANDIDATE, working_width=640)
    assert gray.ndim == 2
    assert color.ndim == 3
    assert color.shape[2] == 3


def test_output_width_matches_working_width() -> None:
    gray, color = load_and_prepare(CANDIDATE, working_width=640)
    assert gray.shape[1] == 640
    assert color.shape[1] == 640


def test_output_dtype_uint8() -> None:
    gray, color = load_and_prepare(CANDIDATE, working_width=640)
    assert gray.dtype == np.uint8
    assert color.dtype == np.uint8


def test_aspect_ratio_preserved() -> None:
    gray, color = load_and_prepare(CANDIDATE, working_width=640)
    # CANDIDATE_ready.jpg is 1920x1080, so at 640 width -> height = 360
    assert 300 < gray.shape[0] < 400


def test_smaller_working_width() -> None:
    gray, color = load_and_prepare(CANDIDATE, working_width=320)
    assert gray.shape[1] == 320
    assert color.shape[1] == 320
