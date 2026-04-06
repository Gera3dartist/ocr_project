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
WARMUP_SECONDS = 0.5
EXPOSURE_SETTLE_SECONDS = 1


def capture(output_path: Path) -> Path:
    """Capture image with LED lighting matching production behavior."""
    gpio.setmode(gpio.BCM)
    gpio.setup(LED_PIN, gpio.OUT)

    cam = Picamera2()
    try:
        gpio.output(LED_PIN, gpio.HIGH)
        time.sleep(WARMUP_SECONDS)

        cam.start()
        time.sleep(EXPOSURE_SETTLE_SECONDS)
        cam.capture_file(str(output_path))
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
