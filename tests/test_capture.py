"""Tests for hardware capture module (LED + camera)."""

from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest

from src.capture import LED_PIN, WARMUP_SECONDS


def test_led_pin_is_gpio2() -> None:
    """LED should use GPIO 2 (lowest usable pin)."""
    assert LED_PIN == 2


def test_warmup_seconds_is_positive() -> None:
    assert WARMUP_SECONDS > 0


@pytest.fixture
def mock_hardware():
    """Provide mock GPIO and Picamera2 injected into capture module."""
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.OUT = 0
    mock_gpio.HIGH = 1
    mock_gpio.LOW = 0

    mock_cam = MagicMock()
    mock_picamera2_cls = MagicMock(return_value=mock_cam)

    with (
        patch("src.capture.GPIO", mock_gpio),
        patch("src.capture.Picamera2", mock_picamera2_cls),
        patch("src.capture.time"),
    ):
        yield mock_gpio, mock_cam, mock_picamera2_cls


def test_capture_turns_led_on_then_off(
    mock_hardware: tuple, tmp_path: Path
) -> None:
    """LED must turn on before capture and off after."""
    mock_gpio, _, _ = mock_hardware
    from src.capture import capture_image

    capture_image(str(tmp_path / "shot.jpg"))

    gpio_calls = mock_gpio.output.call_args_list
    assert call(LED_PIN, mock_gpio.HIGH) in gpio_calls
    assert call(LED_PIN, mock_gpio.LOW) in gpio_calls

    high_idx = gpio_calls.index(call(LED_PIN, mock_gpio.HIGH))
    low_idx = gpio_calls.index(call(LED_PIN, mock_gpio.LOW))
    assert high_idx < low_idx


def test_capture_takes_photo(
    mock_hardware: tuple, tmp_path: Path
) -> None:
    """Camera must capture a file to the given path."""
    _, mock_cam, _ = mock_hardware
    from src.capture import capture_image

    output = str(tmp_path / "shot.jpg")
    result = capture_image(output)

    mock_cam.capture_file.assert_called_once_with(output)
    assert result == output


def test_capture_cleans_up_gpio(
    mock_hardware: tuple, tmp_path: Path
) -> None:
    """GPIO cleanup must be called even on success."""
    mock_gpio, _, _ = mock_hardware
    from src.capture import capture_image

    capture_image(str(tmp_path / "shot.jpg"))
    mock_gpio.cleanup.assert_called_once()


def test_capture_cleans_up_on_camera_error(
    mock_hardware: tuple, tmp_path: Path
) -> None:
    """GPIO cleanup must happen even if camera raises."""
    mock_gpio, mock_cam, _ = mock_hardware
    mock_cam.capture_file.side_effect = RuntimeError("camera error")
    from src.capture import capture_image

    with pytest.raises(RuntimeError, match="camera error"):
        capture_image(str(tmp_path / "shot.jpg"))

    mock_gpio.output.assert_any_call(LED_PIN, mock_gpio.LOW)
    mock_gpio.cleanup.assert_called_once()


def test_capture_stops_camera(
    mock_hardware: tuple, tmp_path: Path
) -> None:
    """Camera must be stopped and closed after capture."""
    _, mock_cam, _ = mock_hardware
    from src.capture import capture_image

    capture_image(str(tmp_path / "shot.jpg"))

    mock_cam.stop.assert_called_once()
    mock_cam.close.assert_called_once()
