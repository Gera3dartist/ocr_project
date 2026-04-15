"""Tests for Flask measurement server."""

from unittest.mock import patch, MagicMock
import json

import pytest

from src.server import create_app
from src.recognizer import MeterReading


@pytest.fixture
def client():
    """Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def mock_pipeline():
    """Mock capture + pipeline to avoid hardware deps."""
    reading = MeterReading(
        digits="03814",
        confidence=[0.12, 0.25, 0.18, 0.09, 0.31],
        transitioning=[False, False, False, True, False],
    )
    with (
        patch("src.server.capture_image", return_value="/tmp/shot.jpg") as mock_cap,
        patch("src.server.read_meter", return_value=reading) as mock_read,
    ):
        yield mock_cap, mock_read, reading


def test_read_endpoint_returns_json(client, mock_pipeline) -> None:
    """GET /read must return JSON with digits and metadata."""
    resp = client.get("/read")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["digits"] == "03814"
    assert len(data["confidence"]) == 5
    assert len(data["transitioning"]) == 5


def test_read_endpoint_triggers_capture(client, mock_pipeline) -> None:
    """Endpoint must call capture_image before reading."""
    mock_cap, mock_read, _ = mock_pipeline
    client.get("/read")
    mock_cap.assert_called_once()
    mock_read.assert_called_once()


def test_read_endpoint_returns_timestamp(client, mock_pipeline) -> None:
    """Response must include an ISO timestamp."""
    resp = client.get("/read")
    data = json.loads(resp.data)
    assert "timestamp" in data
    assert "T" in data["timestamp"]  # ISO format contains T


def test_read_endpoint_handles_capture_error(client) -> None:
    """If capture fails, return 500 with error message."""
    with patch(
        "src.server.capture_image",
        side_effect=RuntimeError("camera disconnected"),
    ):
        resp = client.get("/read")
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert "error" in data


def test_read_endpoint_handles_pipeline_error(client) -> None:
    """If pipeline fails, return 500 with error message."""
    with (
        patch("src.server.capture_image", return_value="/tmp/shot.jpg"),
        patch(
            "src.server.read_meter",
            side_effect=ValueError("Cannot detect counter window"),
        ),
    ):
        resp = client.get("/read")
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert "error" in data


def test_health_endpoint(client) -> None:
    """GET /health returns 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"


# --- /capture endpoint tests ---


def test_capture_endpoint_returns_png(client) -> None:
    """GET /capture must return a PNG image."""
    import numpy as np

    # Mock capture to write a real image, and mock the pipeline stages
    fake_region = np.full((20, 100), 50, dtype=np.uint8)
    with (
        patch("src.server.capture_image", return_value="/tmp/shot.jpg"),
        patch("src.server.load_and_prepare", return_value=(fake_region, fake_region)),
        patch("src.server.find_counter_window", return_value=fake_region),
    ):
        resp = client.get("/capture")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        # PNG files start with the PNG magic bytes
        assert resp.data[:4] == b"\x89PNG"


def test_capture_endpoint_handles_error(client) -> None:
    """If capture fails, return 500 with error."""
    with patch(
        "src.server.capture_image",
        side_effect=RuntimeError("camera disconnected"),
    ):
        resp = client.get("/capture")
        assert resp.status_code == 500
        data = json.loads(resp.data)
        assert "error" in data
