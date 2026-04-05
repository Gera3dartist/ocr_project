import json
import tempfile
from pathlib import Path

import pytest

from src.config import Config, ROI, load_config, save_config


def test_config_defaults() -> None:
    config = Config()
    assert config.num_digits == 5
    assert config.working_width == 640
    assert config.template_size == (40, 60)
    assert config.confidence_threshold == 0.6


def test_roi_defaults() -> None:
    roi = ROI()
    assert 0.0 <= roi.x_norm <= 1.0
    assert 0.0 <= roi.y_norm <= 1.0
    assert roi.w_norm > 0.0
    assert roi.h_norm > 0.0


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    config = Config(
        roi=ROI(x_norm=0.1, y_norm=0.2, w_norm=0.3, h_norm=0.4),
        num_digits=5,
        working_width=320,
        confidence_threshold=0.7,
    )
    path = tmp_path / "config.json"
    save_config(config, str(path))

    loaded = load_config(str(path))
    assert loaded.roi.x_norm == pytest.approx(0.1)
    assert loaded.roi.y_norm == pytest.approx(0.2)
    assert loaded.working_width == 320
    assert loaded.confidence_threshold == pytest.approx(0.7)
    assert loaded.template_size == (40, 60)


def test_save_creates_valid_json(tmp_path: Path) -> None:
    config = Config()
    path = tmp_path / "config.json"
    save_config(config, str(path))

    with open(path) as f:
        data = json.load(f)

    assert "roi" in data
    assert "num_digits" in data
    assert data["num_digits"] == 5
