#!/usr/bin/env python3
"""Capture a test image with LED lighting, mimicking the production pipeline.

Usage:
    python scripts/capture_test_image.py [output_path]

Defaults to saving as capture_YYYYMMDD_HHMMSS.jpg in the current directory.
The image is also printed as raw bytes to stdout when piped (e.g. over SSH):

    ssh gera3000@192.168.0.46 'cd ocr_project && .venv/bin/python scripts/capture_test_image.py -' > local_image.jpg
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import RPi.GPIO as gpio
from picamera2 import Picamera2

LED_PIN = 2
LED_WARMUP = 1.0
AE_SETTLE = 2.0
SETTLE_FRAMES = 3


def capture(output_path: Path) -> Path:
    """Capture image with LED lighting matching production behavior.

    Sequence mirrors src/capture.capture_image() exactly:
        1. LED on, wait for stable brightness
        2. Camera start, wait for auto-exposure convergence
        3. Flush settle frames to lock exposure
        4. Capture final frame
    """
    gpio.setmode(gpio.BCM)
    gpio.setup(LED_PIN, gpio.OUT)

    cam = Picamera2()
    try:
        gpio.output(LED_PIN, gpio.HIGH)
        time.sleep(LED_WARMUP)

        cam.start()
        time.sleep(AE_SETTLE)

        for _ in range(SETTLE_FRAMES):
            cam.capture_array()

        cam.capture_file(str(output_path))

    finally:
        gpio.output(LED_PIN, gpio.LOW)
        cam.stop()
        cam.close()
        gpio.cleanup()

    return output_path


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        # Stream mode: capture to temp file, write bytes to stdout
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            capture(tmp_path)
            sys.stdout.buffer.write(tmp_path.read_bytes())
        return

    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"capture_{timestamp}.jpg")

    capture(output_path)
    print(f"Saved: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
