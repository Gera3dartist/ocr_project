# Gas Meter OCR —  Digit Reader

Computer-vision pipeline that reads the 5-digit mechanical display on a SAMGAZ G4 gas meter. Designed to run on a **Raspberry Pi Zero W2** with a camera module, but can also process static images on any machine.

## How It Works

1. **Capture** — triggers the RPi camera (with LED illumination)
2. **Preprocess** — resizes, applies CLAHE contrast enhancement and Gaussian blur
3. **ROI Detection** — locates the counter window and separates black/red digit regions
4. **Segmentation** — splits the region into 5 individual digit images
5. **Recognition** — template matching with hybrid scoring (correlation + pixel overlap), including transitioning-digit detection

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### CLI

```bash
python -m src.pipeline --image path/to/meter.jpg
```

### REST API

```bash
python -m src.server          # starts Flask on port 5000
curl http://localhost:5000/read
```

## Project Structure

```
src/
  pipeline.py        # orchestrates the full reading flow
  server.py          # Flask API (/read endpoint)
  capture.py         # RPi camera & LED control
  preprocessing.py   # image loading & enhancement
  roi_detector.py    # counter-window detection
  segmenter.py       # digit isolation
  recognizer.py      # template matching engine
  config.py          # JSON-based configuration
templates/           # pre-built digit templates (0-9)
tools/               # template builder utility
tests/               # pytest suite
```

## Configuration

All tuneable parameters (ROI coordinates, thresholds, template size) live in `config.json`.

## Tech Stack

Python 3.13+ · OpenCV · NumPy · Flask · picamera2 · RPi.GPIO
