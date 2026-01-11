# FSAE Scraper

A professional desktop application for extracting telemetry data from racing/motorsport onboard videos using advanced OCR and computer vision.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## ✨ Features

- **🚀 Advanced OCR Engine** - Uses PaddleOCR (with EasyOCR fallback) for maximum accuracy across different fonts and overlays
- **📊 Multiple Metrics** - Extract speed, G-force, per-wheel torque, and more
- **🎯 Interactive ROI Selection** - Draw regions of interest directly on the video
- **⚡ Real-time Preview** - See extracted values overlaid on video playback
- **🎨 Modern UI** - Polished dark theme with professional styling
- **📦 Standalone Executable** - Distribute to others without requiring Python

## 🖼️ Screenshots

*Coming soon*

## 📥 Installation

### Option 1: Download Executable (Easiest)

1. Download the latest release from the [Releases](../../releases/latest) page
2. Run `FSAE_Scraper_vX.X.X.exe` (version number will be in the filename)
3. No Python installation required!

> 💡 **Note:** The executable is automatically built and released on every commit to master.

#### 🎮 GPU Acceleration (Optional)

The executable supports **NVIDIA GPU acceleration** for faster OCR processing:

| Setup | Performance |
|-------|-------------|
| **With NVIDIA GPU + CUDA** | ⚡ Fast - Uses GPU automatically |
| **Without GPU** | ✅ Works fine - Falls back to CPU |

**To enable GPU acceleration:**
1. Have an NVIDIA GPU (GTX 10 series or newer recommended)
2. Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
3. That's it! The exe will detect CUDA and use your GPU automatically

> The exe works without CUDA - it's just slower. No extra setup needed for CPU-only usage.

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/fsae_data_extractor.git
cd fsae_data_extractor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### GPU Support (Source Installation)

For GPU acceleration when running from source:

```bash
# Instead of paddlepaddle, install the GPU version:
pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# For EasyOCR GPU support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Note for PaddleOCR on Windows:
If you encounter issues installing PaddleOCR, try:
```bash
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr
```

## 🚀 Usage

### Running the Application

```bash
python run.py
```

### Quick Start

1. **Open a Video** - File → Open Video (or Ctrl+O)
2. **Initialize OCR** - Tools → Initialize OCR (first time only, will auto-download models)
3. **Select ROIs** - Click "Select ROI" for each metric and draw a box around the data area
4. **Play Video** - Use transport controls or Space to play/pause
5. **Export Data** - File → Export Data to save extracted values to CSV

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| ← → | Step 1 frame |
| ↑ ↓ | Step 10 frames |
| Ctrl+O | Open video |
| Ctrl+S | Save configuration |
| Ctrl+E | Export data |

## 🏗️ Building Standalone Executable

To create a distributable executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
python build.py

# Or with options
python build.py --clean    # Clean first
python build.py --onedir   # Build as folder (easier to debug)
```

The executable will be in the `dist/` folder.

## 📁 Project Structure

```
fsae_data_extractor/
├── run.py                  # Application entry point
├── build.py                # Build script for executable
├── requirements.txt        # Python dependencies
├── fsae_extractor.spec     # PyInstaller configuration
├── src/
│   ├── core/
│   │   ├── ocr_engine.py   # PaddleOCR/EasyOCR wrapper
│   │   ├── preprocessor.py # Image preprocessing pipeline
│   │   ├── video.py        # Video handling
│   │   └── extractors/     # Metric extractors
│   │       ├── base.py     # Base extractor class
│   │       ├── speed.py    # Speed OCR extractor
│   │       ├── gforce.py   # G-force OCR extractor
│   │       └── torque.py   # Torque color analyzer
│   ├── gui/
│   │   ├── main_window.py  # Main application window
│   │   ├── styles.py       # Modern UI styling
│   │   ├── splash.py       # Splash screen
│   │   └── widgets/        # Custom UI widgets
│   └── config/
│       └── settings.py     # Configuration management
├── assets/                 # App icons and images
└── dist/                   # Built executables (after build)
```

## ⚙️ Configuration

### Preprocessing Presets

The app includes optimized presets for different video types:

| Preset | Best For |
|--------|----------|
| `racing_hud` | Racing game/sim overlays (default) |
| `f1_tv` | Official F1 TV broadcasts |
| `digital_display` | LCD/digital dashboard displays |
| `minimal` | Clean overlays needing little processing |
| `aggressive` | Noisy or low-quality video |

### OCR Engine Selection

- **PaddleOCR** (default) - Best accuracy, especially for varied fonts
- **EasyOCR** - Simpler installation, used as fallback

The app automatically selects PaddleOCR if available, falling back to EasyOCR.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.
