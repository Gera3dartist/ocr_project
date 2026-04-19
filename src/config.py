"""Configuration for SAMGAZ G4 gas meter OCR pipeline."""

import json
from dataclasses import dataclass, field, asdict


@dataclass
class ROI:
    """Region of interest in normalized coordinates (0.0-1.0)."""

    x_norm: float = 0.22
    y_norm: float = 0.52
    w_norm: float = 0.55
    h_norm: float = 0.15


@dataclass
class Config:
    """Pipeline configuration."""

    roi: ROI = field(default_factory=ROI)
    num_digits: int = 5
    template_size: tuple[int, int] = (40, 60)
    working_width: int = 640
    confidence_threshold: float = 0.6
    transition_margin: float = 0.1
    gsheet_file_name: str = 'sensor_readings'


def save_config(config: Config, path: str) -> None:
    """Save config to JSON file."""
    data = asdict(config)
    data["template_size"] = list(config.template_size)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: str) -> Config:
    """Load config from JSON file."""
    with open(path) as f:
        data = json.load(f)

    roi_data = data.pop("roi", {})
    roi = ROI(**roi_data)
    data["roi"] = roi
    data["template_size"] = tuple(data["template_size"])
    return Config(**data)
