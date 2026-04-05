"""Hardware capture: LED control + camera for RPi Zero W2.

Imports RPi.GPIO and picamera2 at call time so the module
can be imported (and tested with mocks) on non-RPi hosts.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

LED_PIN = 2
WARMUP_SECONDS = 0.5


def _import_gpio() -> Any:
    return importlib.import_module("RPi.GPIO")


def _import_picamera2() -> Any:
    return importlib.import_module("picamera2").Picamera2


# Allow patching in tests
GPIO: Any = None
Picamera2: Any = None


def capture_image(output_path: str) -> str:
    """Turn on LED, capture image, turn off LED.

    Args:
        output_path: File path to save the captured JPEG.

    Returns:
        The output_path for convenience.

    Raises:
        RuntimeError: If camera capture fails.
    """
    gpio = GPIO if GPIO is not None else _import_gpio()
    picamera2_cls = Picamera2 if Picamera2 is not None else _import_picamera2()

    gpio.setmode(gpio.BCM)
    gpio.setup(LED_PIN, gpio.OUT)

    cam = picamera2_cls()
    try:
        gpio.output(LED_PIN, gpio.HIGH)
        time.sleep(WARMUP_SECONDS)

        cam.start()
        time.sleep(1)  # auto-exposure settle
        cam.capture_file(output_path)
        cam.stop()
        cam.close()

        gpio.output(LED_PIN, gpio.LOW)
    except Exception:
        gpio.output(LED_PIN, gpio.LOW)
        cam.stop()
        cam.close()
        raise
    finally:
        gpio.cleanup()

    return output_path
