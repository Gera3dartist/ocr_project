"""Flask server for gas meter reading endpoint."""

from __future__ import annotations

import io
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, jsonify, send_file

from src.capture import capture_image
from src.config import load_config
from src.pipeline import read_meter
from src.preprocessing import load_and_prepare
from src.roi_detector import find_counter_window
from src.segmenter import segment_digits

PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.get("/read")
    def read_measurement():
        """Capture image and run OCR pipeline.

        Returns:
            JSON with digits, confidence, transitioning, and timestamp.
        """
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                image_path = tmp.name

            capture_image(image_path)
            reading = read_meter(image_path)

            return jsonify(
                {
                    "digits": reading.digits,
                    "confidence": reading.confidence,
                    "transitioning": reading.transitioning,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as e:
            logger.exception("Measurement failed")
            return jsonify({"error": str(e)}), 500

    @app.get("/capture")
    def capture_debug():
        """Capture image and return the cropped, binarized digit region as PNG.

        Use this to visually verify what the pipeline sees — a quick way to
        check camera alignment, lighting, and digit quality without running
        full recognition.

        Returns:
            PNG image of the binarized digit region (white digits on black).
        """
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp:
                image_path = tmp.name

            capture_image(image_path)

            config = load_config(str(PROJECT_ROOT / "config.json"))
            gray, color = load_and_prepare(image_path, config.working_width)
            black_region = find_counter_window(gray, color, config)

            # Binarize (same as segmenter does)
            _, binary = cv2.threshold(
                black_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            _, png_bytes = cv2.imencode(".png", binary)
            return send_file(
                io.BytesIO(png_bytes.tobytes()),
                mimetype="image/png",
            )
        except Exception as e:
            logger.exception("Capture failed")
            return jsonify({"error": str(e)}), 500

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
