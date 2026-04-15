"""Hardware capture: LED control + camera for RPi Zero W2.

Imports RPi.GPIO and picamera2 at call time so the module
can be imported (and tested with mocks) on non-RPi hosts.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

LED_PIN = 2

# LED needs ~1s to reach full, stable brightness.
LED_WARMUP = 1.0

# Camera auto-exposure needs 2s after start() to converge.
# The first few frames have unstable gain/exposure — we discard them.
AE_SETTLE = 2.0

# Number of throwaway frames before the real capture.
# Each frame lets the AE algorithm refine; by the 3rd frame the
# exposure is locked to the LED-lit scene.
SETTLE_FRAMES = 3


def _import_gpio() -> Any:
    return importlib.import_module("RPi.GPIO")


def _import_picamera2() -> Any:
    return importlib.import_module("picamera2").Picamera2


# Allow patching in tests
GPIO: Any = None
Picamera2: Any = None


def capture_image(output_path: str) -> str:
    """Turn on LED, wait for stable lighting + exposure, capture image.

    Sequence:
        1. LED on → wait LED_WARMUP for stable brightness
        2. Camera start → wait AE_SETTLE for auto-exposure convergence
        3. Flush SETTLE_FRAMES to lock exposure to the lit scene
        4. Capture final frame
        5. Cleanup (LED off, camera closed, GPIO released)

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
        # 1. LED on and warm up
        gpio.output(LED_PIN, gpio.HIGH)
        time.sleep(LED_WARMUP)

        # 2. Start camera, let auto-exposure converge under LED light
        cam.start()
        time.sleep(AE_SETTLE)

        # 3. Flush initial frames so AE is fully locked
        for _ in range(SETTLE_FRAMES):
            cam.capture_array()

        # 4. Capture the actual frame
        cam.capture_file(output_path)

    finally:
        gpio.output(LED_PIN, gpio.LOW)
        cam.stop()
        cam.close()
        gpio.cleanup()

    return output_path
