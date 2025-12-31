# FSAE Onboard Video Data Extractor

A Python application for extracting telemetry data from Formula SAE onboard videos using OCR and computer vision.

## Features

- **Speed Extraction** - OCR-based speed reading from video overlays
- **G-Force Extraction** - OCR-based lateral/longitudinal G readings
- **Torque Analysis** - Color-based per-wheel torque visualization (green=drive, red=brake)
- **Interactive ROI Selection** - GUI for defining regions of interest for each metric
- **Real-time Preview** - See extracted values overlaid on video

## Installation

```bash
# Clone and navigate
cd fsae_data_extractor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
fsae_data_extractor/
├── src/
│   ├── core/               # Core extraction logic
│   │   ├── extractors/     # Individual metric extractors
│   │   │   ├── base.py     # Base extractor class
│   │   │   ├── speed.py    # Speed OCR extractor
│   │   │   ├── gforce.py   # G-force OCR extractor
│   │   │   └── torque.py   # Torque color analyzer
│   │   ├── video.py        # Video handling utilities
│   │   └── roi.py          # ROI management
│   ├── gui/                # PyQt6 frontend
│   │   ├── main_window.py  # Main application window
│   │   ├── widgets/        # Custom widgets
│   │   └── dialogs/        # Configuration dialogs
│   └── config/             # Configuration management
├── tests/                  # Unit tests
├── data/                   # Sample data and outputs
├── config.json             # User configuration
└── requirements.txt
```

## Usage

### GUI Mode (Recommended)
```bash
python -m src.gui.main_window
```

### CLI Mode
```bash
# Extract all metrics
python -m src.core.extract --video path/to/video.mp4 --output data/results.csv

# Extract specific metric
python -m src.core.extract --video path/to/video.mp4 --metric speed
```

## Metrics Configuration

Each metric can be individually configured with:
- **ROI** (Region of Interest) - The area of the video frame to analyze
- **Extraction Method** - OCR or color analysis
- **Thresholds** - HSV ranges for color detection, confidence for OCR

## License

MIT
