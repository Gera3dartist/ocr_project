"""Flask server for gas meter reading endpoint."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify

from src.capture import capture_image
from src.pipeline import read_meter

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

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
