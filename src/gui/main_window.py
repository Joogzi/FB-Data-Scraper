"""Main application window for FB Data Scraper."""

import sys
import os
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QAction, QKeySequence, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QStatusBar, QMenuBar,
    QMenu, QToolBar, QLabel, QProgressBar
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.video import VideoReader
from src.core.extractors.base import ROI, ExtractionResult
from src.core.extractors.speed import SpeedExtractor
from src.core.extractors.gforce import GForceExtractor
from src.core.extractors.kilowatt import KilowattExtractor
from src.core.extractors.torque import FourWheelTorqueExtractor, TorqueResult, PedalExtractor
from src.core.ocr_engine import cleanup_shared_engine
from src.config.settings import ConfigManager, MetricConfig
from src.gui.widgets.video_widget import VideoWidget
from src.gui.widgets.metric_panel import MetricPanel
from src.gui.widgets.transport import TransportControls


class OCRInitWorker(QObject):
    """Worker for initializing OCR in a background thread."""
    finished = pyqtSignal(bool, str)  # cuda_available, error_msg
    status = pyqtSignal(str)  # Status updates
    
    def __init__(self, speed_extractor, gforce_extractor, kw_extractor):
        super().__init__()
        self.speed_extractor = speed_extractor
        self.gforce_extractor = gforce_extractor
        self.kw_extractor = kw_extractor
    
    def run(self):
        """Initialize OCR engines in background."""
        from src.core.ocr_engine import OCRBackend, cleanup_shared_engine
        
        try:
            self.status.emit("Cleaning up previous OCR engine...")
            cleanup_shared_engine()
            
            # Check for CUDA availability
            cuda_available = False
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except ImportError:
                pass
            
            self.status.emit("Setting up EasyOCR extractors...")
            
            # Clear and set backend for extractors
            self.speed_extractor._ocr_engine = None
            self.gforce_extractor._ocr_engine = None
            self.kw_extractor._ocr_engine = None
            
            self.speed_extractor.ocr_backend = OCRBackend.EASYOCR
            self.gforce_extractor.ocr_backend = OCRBackend.EASYOCR
            self.kw_extractor.ocr_backend = OCRBackend.EASYOCR
            
            self.speed_extractor.use_gpu = cuda_available
            self.gforce_extractor.use_gpu = cuda_available
            self.kw_extractor.use_gpu = cuda_available
            
            self.status.emit("Initializing EasyOCR (downloading models if needed)...")
            
            self.speed_extractor.initialize()
            self.gforce_extractor.initialize()
            self.kw_extractor.initialize()
            
            self.finished.emit(cuda_available, "")
            
        except Exception as e:
            self.finished.emit(False, str(e))


class ExportWorker(QObject):
    """Worker for batch extracting data from video frames."""
    progress = pyqtSignal(int, int)  # current_frame, total_frames
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, video_reader, extractors: dict, output_path: str, metric_panel):
        super().__init__()
        self.video_reader = video_reader
        self.extractors = extractors
        self.output_path = output_path
        self.metric_panel = metric_panel
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        """Extract data from all frames and save to CSV."""
        import csv
        
        try:
            total_frames = self.video_reader.frame_count
            fps = self.video_reader.fps
            
            # Determine which metrics have valid ROIs and are enabled
            active_metrics = []
            headers = ["Frame", "Time (s)"]
            
            # Check each extractor
            if self.extractors.get('speed') and self.extractors['speed'].roi:
                if self.metric_panel.speed_card.is_enabled:
                    active_metrics.append(('speed', self.extractors['speed']))
                    headers.append("Speed (km/h)")
            
            if self.extractors.get('gforce') and self.extractors['gforce'].roi:
                if self.metric_panel.gforce_card.is_enabled:
                    active_metrics.append(('gforce', self.extractors['gforce']))
                    headers.append("G-Force")
            
            if self.extractors.get('kw') and self.extractors['kw'].roi:
                if hasattr(self.metric_panel, 'kw_card') and self.metric_panel.kw_card.is_enabled:
                    active_metrics.append(('kw', self.extractors['kw']))
                    headers.append("Power (kW)")
            
            if self.extractors.get('throttle') and self.extractors['throttle'].roi:
                if hasattr(self.metric_panel, 'throttle_card') and self.metric_panel.throttle_card.is_enabled:
                    active_metrics.append(('throttle', self.extractors['throttle']))
                    headers.append("Throttle (%)")
            
            if self.extractors.get('brake') and self.extractors['brake'].roi:
                if hasattr(self.metric_panel, 'brake_card') and self.metric_panel.brake_card.is_enabled:
                    active_metrics.append(('brake', self.extractors['brake']))
                    headers.append("Brake (%)")
            
            # Torque metrics
            if self.extractors.get('torque'):
                for wheel in ['fl', 'fr', 'rl', 'rr']:
                    roi = self.extractors['torque'].get_roi(wheel)
                    card = self.metric_panel.torque_cards.get(wheel)
                    if roi and card and card.is_enabled:
                        active_metrics.append((f'torque_{wheel}', (self.extractors['torque'], wheel)))
                        wheel_name = {'fl': 'FL', 'fr': 'FR', 'rl': 'RL', 'rr': 'RR'}[wheel]
                        headers.append(f"Torque {wheel_name}")
            
            if not active_metrics:
                self.finished.emit(False, "No metrics with valid ROIs selected. Please draw ROI boxes for the metrics you want to extract.")
                return
            
            # Open CSV file
            with open(self.output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                # Process each frame
                for frame_idx in range(total_frames):
                    if self._cancelled:
                        self.finished.emit(False, "Export cancelled")
                        return
                    
                    self.progress.emit(frame_idx + 1, total_frames)
                    
                    ret, frame = self.video_reader.read_frame(frame_idx)
                    if not ret or frame is None:
                        continue
                    
                    # Calculate time
                    time_sec = frame_idx / fps if fps > 0 else 0
                    row = [frame_idx, f"{time_sec:.3f}"]
                    
                    # Extract each active metric
                    for metric_name, extractor in active_metrics:
                        value = ""
                        try:
                            if metric_name.startswith('torque_'):
                                torque_ext, wheel = extractor
                                result = torque_ext.extractors[wheel].extract_from_roi(frame)
                                if result.is_valid:
                                    value = f"{result.value.value:.2f}"
                            elif metric_name in ('throttle', 'brake'):
                                result = extractor.extract_from_roi(frame)
                                if result.is_valid:
                                    value = f"{result.value * 100:.1f}"
                            else:
                                result = extractor.extract_from_roi(frame)
                                if result.is_valid:
                                    if metric_name == 'gforce':
                                        value = f"{result.value:.2f}"
                                    elif metric_name == 'kw':
                                        value = f"{result.value:.1f}"
                                    else:
                                        value = str(result.value)
                        except Exception as e:
                            print(f"Error extracting {metric_name} at frame {frame_idx}: {e}")
                        
                        row.append(value)
                    
                    writer.writerow(row)
            
            self.finished.emit(True, f"Successfully exported {total_frames} frames to {self.output_path}")
            
        except Exception as e:
            self.finished.emit(False, f"Export failed: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("FB Data Scraper")
        self.setMinimumSize(1200, 800)
        
        # Core components
        self.video_reader: Optional[VideoReader] = None
        self.config_manager = ConfigManager()
        
        # Extractors
        self.speed_extractor = SpeedExtractor()
        self.gforce_extractor = GForceExtractor()
        self.kw_extractor = KilowattExtractor()
        self.torque_extractor = FourWheelTorqueExtractor()
        self.throttle_extractor = PedalExtractor(pedal_type="accelerator")
        self.brake_extractor = PedalExtractor(pedal_type="brake")
        
        # Playback state
        self._current_frame_idx = 0
        self._is_playing = False
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._on_play_tick)
        
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._connect_signals()
        
        # Load last config
        self._load_config()
    
    def _setup_ui(self):
        """Set up the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Video display and controls
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_widget = VideoWidget()
        video_layout.addWidget(self.video_widget, 1)
        
        self.transport = TransportControls()
        video_layout.addWidget(self.transport)
        
        splitter.addWidget(video_panel)
        
        # Right: Metric configuration panel
        self.metric_panel = MetricPanel()
        splitter.addWidget(self.metric_panel)
        
        # Set initial sizes (70% video, 30% panel)
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # OCR backend indicator
        self.ocr_label = QLabel("OCR: Initializing...")
        self.ocr_label.setStyleSheet("color: #888; padding-right: 10px;")
        self.status_bar.addPermanentWidget(self.ocr_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def _setup_menu(self):
        """Set up menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_video_dialog)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_config_action = QAction("&Save Configuration", self)
        save_config_action.setShortcut(QKeySequence.StandardKey.Save)
        save_config_action.triggered.connect(self._save_config)
        file_menu.addAction(save_config_action)
        
        load_config_action = QAction("&Load Configuration...", self)
        load_config_action.triggered.connect(self._load_config_dialog)
        file_menu.addAction(load_config_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Data...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        # OCR re-initialization option
        init_ocr_action = QAction("Re-initialize &OCR", self)
        init_ocr_action.triggered.connect(self._initialize_ocr)
        tools_menu.addAction(init_ocr_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # These will work globally in the window
        pass  # Handled in keyPressEvent
    
    def _connect_signals(self):
        """Connect widget signals."""
        # Video widget
        self.video_widget.roi_selected.connect(self._on_roi_selected)
        
        # Metric panel
        self.metric_panel.select_roi_requested.connect(self._on_select_roi_requested)
        
        # Transport controls
        self.transport.play_clicked.connect(self._play)
        self.transport.pause_clicked.connect(self._pause)
        self.transport.seek_requested.connect(self._seek)
        self.transport.step_forward.connect(lambda n: self._step(n))
        self.transport.step_backward.connect(lambda n: self._step(-n))
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()
        
        if key == Qt.Key.Key_Space:
            if self._is_playing:
                self._pause()
            else:
                self._play()
            event.accept()
        elif key == Qt.Key.Key_Right:
            self._step(1)  # 1 frame forward
            event.accept()
        elif key == Qt.Key.Key_Left:
            self._step(-1)  # 1 frame backward
            event.accept()
        elif key == Qt.Key.Key_Up:
            self._step(10)  # 10 frames forward
            event.accept()
        elif key == Qt.Key.Key_Down:
            self._step(-10)  # 10 frames backward
            event.accept()
        else:
            super().keyPressEvent(event)
    
    # --- Video handling ---
    
    def _open_video_dialog(self):
        """Show file dialog to open video."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if path:
            self._open_video(path)
    
    def _open_video(self, path: str):
        """Open a video file."""
        # Close existing
        if self.video_reader:
            self.video_reader.close()
        
        self.video_reader = VideoReader(path)
        if not self.video_reader.open():
            QMessageBox.critical(self, "Error", f"Failed to open video: {path}")
            return
        
        info = self.video_reader.info
        self.transport.set_video_info(info.frame_count, info.fps)
        
        # Update config
        self.config_manager.update(video_path=path)
        
        # Show first frame
        self._seek(0)
        
        self.status_label.setText(f"Loaded: {Path(path).name}")
    
    def _display_frame(self, frame_idx: int):
        """Display a specific frame and run extractions."""
        if self.video_reader is None:
            return
        
        ret, frame = self.video_reader.read_frame(frame_idx)
        if not ret or frame is None:
            return
        
        self._current_frame_idx = frame_idx
        self.video_widget.set_frame(frame)
        self.transport.set_current_frame(frame_idx)
        
        # Run extractions
        self._run_extractions(frame)
    
    def _run_extractions(self, frame):
        """Run all enabled extractors on the current frame."""
        # ===== Extract pedals FIRST (needed for kW sign correction) =====
        throttle_pct = 0.0
        brake_pct = 0.0
        
        # Throttle pedal
        if self.throttle_extractor.roi and hasattr(self.metric_panel, 'throttle_card') and self.metric_panel.throttle_card.is_enabled:
            result = self.throttle_extractor.extract_from_roi(frame)
            if result.is_valid:
                throttle_pct = result.value * 100
                self.metric_panel.set_value("throttle", f"{throttle_pct:.0f}%")
                self.video_widget.update_roi_value("throttle", f"{throttle_pct:.0f}%")
            else:
                self.metric_panel.set_value("throttle", "--")
        
        # Brake pedal
        if self.brake_extractor.roi and hasattr(self.metric_panel, 'brake_card') and self.metric_panel.brake_card.is_enabled:
            result = self.brake_extractor.extract_from_roi(frame)
            if result.is_valid:
                brake_pct = result.value * 100
                # If throttle is significantly pressed, brake should be 0
                # (unless brake value is very high, indicating actual braking)
                if throttle_pct > 10 and brake_pct < 30:
                    brake_pct = 0.0
                self.metric_panel.set_value("brake", f"{brake_pct:.0f}%")
                self.video_widget.update_roi_value("brake", f"{brake_pct:.0f}%")
            else:
                self.metric_panel.set_value("brake", "--")
        
        # ===== Speed =====
        speed_value = 0.0
        if self.speed_extractor.roi and self.metric_panel.speed_card.is_enabled:
            result = self.speed_extractor.extract_from_roi(frame)
            if result.is_valid:
                speed_value = float(result.value)
                self.metric_panel.set_value("speed", f"{result.value} km/h")
                self.video_widget.update_roi_value("speed", f"{result.value}")
            else:
                self.metric_panel.set_value("speed", "--")
        
        # ===== G-force (with decimals) =====
        if self.gforce_extractor.roi and self.metric_panel.gforce_card.is_enabled:
            result = self.gforce_extractor.extract_from_roi(frame)
            if result.is_valid:
                self.metric_panel.set_value("gforce", f"{result.value:.2f} G")
                self.video_widget.update_roi_value("gforce", f"{result.value:.2f}G")
            else:
                self.metric_panel.set_value("gforce", "--")
        
        # ===== Kilowatts (uses speed + brake for sign correction) =====
        if self.kw_extractor.roi and hasattr(self.metric_panel, 'kw_card') and self.metric_panel.kw_card.is_enabled:
            # Pass speed and brake data for sign correction
            # Rule: if speed > 10 km/h and brake is on, kW must be negative
            self.kw_extractor.speed = speed_value
            self.kw_extractor.brake_pct = brake_pct
            
            result = self.kw_extractor.extract_from_roi(frame)
            if result.is_valid:
                self.metric_panel.set_value("kw", f"{result.value:.1f} kW")
                self.video_widget.update_roi_value("kw", f"{result.value:.1f}")
            else:
                self.metric_panel.set_value("kw", "--")
        
        # Torque (all wheels)
        results = self.torque_extractor.extract_all(frame)
        for wheel, result in results.items():
            card_name = f"torque_{wheel}"
            if result.is_valid and isinstance(result.value, TorqueResult):
                val = result.value.value
                self.metric_panel.set_value(card_name, f"{val:+.2f}")
                color = QColor(0, 200, 0) if val >= 0 else QColor(200, 0, 0)
                self.video_widget.update_roi_value(wheel.upper(), f"{val:+.2f}")
    
    # --- Playback controls ---
    
    def _play(self):
        """Start playback."""
        if self.video_reader is None:
            return
        
        self._is_playing = True
        self.transport.set_playing(True)
        
        # Calculate timer interval from FPS
        interval = int(1000 / self.video_reader.fps)
        self._play_timer.start(interval)
    
    def _pause(self):
        """Pause playback."""
        self._is_playing = False
        self.transport.set_playing(False)
        self._play_timer.stop()
    
    def _seek(self, frame_idx: int):
        """Seek to a specific frame."""
        if self.video_reader is None:
            return
        
        frame_idx = max(0, min(frame_idx, self.video_reader.frame_count - 1))
        self._display_frame(frame_idx)
    
    def _step(self, delta: int):
        """Step forward/backward by delta frames."""
        self._seek(self._current_frame_idx + delta)
    
    def _on_play_tick(self):
        """Called on each playback timer tick."""
        if self.video_reader is None:
            self._pause()
            return
        
        next_frame = self._current_frame_idx + 1
        if next_frame >= self.video_reader.frame_count:
            self._pause()
            return
        
        self._display_frame(next_frame)
    
    # --- ROI selection ---
    
    def _on_select_roi_requested(self, metric_name: str):
        """Handle request to select ROI for a metric."""
        if self.video_reader is None:
            QMessageBox.warning(self, "No Video", "Please open a video first.")
            return
        
        self.video_widget.start_roi_selection(metric_name)
        self.status_label.setText(f"Draw ROI for: {metric_name}")
    
    def _on_roi_selected(self, metric_name: str, roi_tuple: tuple):
        """Handle ROI selection completion."""
        x1, y1, x2, y2 = roi_tuple
        roi = ROI(x1=x1, y1=y1, x2=x2, y2=y2, name=metric_name)
        
        # Update the appropriate extractor
        if metric_name == "speed":
            self.speed_extractor.roi = roi
        elif metric_name == "gforce":
            self.gforce_extractor.roi = roi
        elif metric_name == "kw":
            self.kw_extractor.roi = roi
        elif metric_name == "throttle":
            self.throttle_extractor.roi = roi
        elif metric_name == "brake":
            self.brake_extractor.roi = roi
        elif metric_name.startswith("torque_"):
            wheel = metric_name.replace("torque_", "")
            self.torque_extractor.set_roi(wheel, roi)
        
        # Update UI
        self.metric_panel.set_roi(metric_name, x1, y1, x2, y2)
        
        # Set display color based on metric type
        if metric_name == "speed":
            color = QColor(0, 200, 255)  # Cyan
        elif metric_name == "gforce":
            color = QColor(255, 200, 0)  # Gold
        elif metric_name == "kw":
            color = QColor(255, 100, 255)  # Magenta
        elif metric_name == "throttle":
            color = QColor(0, 255, 100)  # Green
        elif metric_name == "brake":
            color = QColor(255, 80, 80)  # Red
        else:
            color = QColor(0, 255, 0)  # Default green for torque
        
        self.video_widget.set_roi(metric_name, roi_tuple, color)
        
        self.status_label.setText(f"ROI set for: {metric_name}")
        
        # Re-run extraction on current frame
        if self.video_reader:
            ret, frame = self.video_reader.read_frame(self._current_frame_idx)
            if ret and frame is not None:
                self._run_extractions(frame)
    
    # --- Configuration ---
    
    def _save_config(self):
        """Save current configuration."""
        # Update config with current ROIs
        if self.speed_extractor.roi:
            self.config_manager.config.speed.roi = self.speed_extractor.roi.to_dict()
        if self.gforce_extractor.roi:
            self.config_manager.config.gforce.roi = self.gforce_extractor.roi.to_dict()
        
        # Save torque ROIs
        for wheel in ["fl", "fr", "rl", "rr"]:
            roi = self.torque_extractor.get_roi(wheel)
            if roi:
                metric_config = getattr(self.config_manager.config, f"torque_{wheel}")
                metric_config.roi = roi.to_dict()
        
        if self.config_manager.save():
            self.status_label.setText("Configuration saved")
        else:
            QMessageBox.warning(self, "Error", "Failed to save configuration")
    
    def _load_config(self):
        """Load configuration from file."""
        config = self.config_manager.load()
        
        # Apply ROIs
        if config.speed.roi:
            roi = ROI.from_dict(config.speed.roi)
            self.speed_extractor.roi = roi
            self.video_widget.set_roi("speed", (roi.x1, roi.y1, roi.x2, roi.y2), QColor(0, 200, 255))
            self.metric_panel.set_roi("speed", roi.x1, roi.y1, roi.x2, roi.y2)
        
        if config.gforce.roi:
            roi = ROI.from_dict(config.gforce.roi)
            self.gforce_extractor.roi = roi
            self.video_widget.set_roi("gforce", (roi.x1, roi.y1, roi.x2, roi.y2), QColor(255, 200, 0))
            self.metric_panel.set_roi("gforce", roi.x1, roi.y1, roi.x2, roi.y2)
        
        for wheel in ["fl", "fr", "rl", "rr"]:
            metric_config = getattr(config, f"torque_{wheel}")
            if metric_config.roi:
                roi = ROI.from_dict(metric_config.roi)
                self.torque_extractor.set_roi(wheel, roi)
                self.video_widget.set_roi(f"torque_{wheel}", (roi.x1, roi.y1, roi.x2, roi.y2), QColor(0, 255, 0))
                self.metric_panel.set_roi(f"torque_{wheel}", roi.x1, roi.y1, roi.x2, roi.y2)
        
        # Open video if configured
        if config.video_path and os.path.exists(config.video_path):
            self._open_video(config.video_path)
    
    def _load_config_dialog(self):
        """Show dialog to load configuration file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            "",
            "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.config_manager = ConfigManager(path)
            self._load_config()
    
    # --- OCR initialization ---
    
    def _initialize_ocr(self):
        """Initialize EasyOCR engine in a background thread."""
        # Show loading state
        self.status_label.setText("Initializing EasyOCR...")
        self.ocr_label.setText("OCR: Initializing...")
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Create worker and thread
        self._ocr_thread = QThread()
        self._ocr_worker = OCRInitWorker(
            self.speed_extractor,
            self.gforce_extractor,
            self.kw_extractor
        )
        self._ocr_worker.moveToThread(self._ocr_thread)
        
        # Connect signals
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.status.connect(self._on_ocr_status)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_worker.finished.connect(self._ocr_worker.deleteLater)
        self._ocr_thread.finished.connect(self._ocr_thread.deleteLater)
        
        # Start the thread
        self._ocr_thread.start()
    
    def _on_ocr_status(self, status: str):
        """Update status label during OCR initialization."""
        self.status_label.setText(status)
    
    def _on_ocr_finished(self, cuda_available: bool, error_msg: str):
        """Handle OCR initialization completion."""
        self.progress_bar.hide()
        
        if error_msg:
            QMessageBox.critical(self, "Error", f"Failed to initialize OCR: {error_msg}")
            self.status_label.setText("OCR initialization failed")
            self.ocr_label.setText("OCR: Failed")
            self.ocr_label.setStyleSheet("color: #F44336; padding-right: 10px;")
            return
        
        # Get GPU info for display
        gpu_status = " (CUDA)" if cuda_available else " (CPU)"
        self.ocr_label.setText(f"OCR: EASYOCR{gpu_status}")
        self.ocr_label.setStyleSheet("color: #4CAF50; padding-right: 10px; font-weight: bold;")
        self.status_label.setText("OCR ready")
    
    # --- Export ---
    
    def _export_data(self):
        """Export extracted data to CSV."""
        if self.video_reader is None:
            QMessageBox.warning(self, "No Video", "Please open a video first.")
            return
        
        # Check if any metrics are selected
        has_metrics = False
        if self.speed_extractor.roi and self.metric_panel.speed_card.is_enabled:
            has_metrics = True
        if self.gforce_extractor.roi and self.metric_panel.gforce_card.is_enabled:
            has_metrics = True
        if self.kw_extractor.roi and hasattr(self.metric_panel, 'kw_card') and self.metric_panel.kw_card.is_enabled:
            has_metrics = True
        if self.throttle_extractor.roi and hasattr(self.metric_panel, 'throttle_card') and self.metric_panel.throttle_card.is_enabled:
            has_metrics = True
        if self.brake_extractor.roi and hasattr(self.metric_panel, 'brake_card') and self.metric_panel.brake_card.is_enabled:
            has_metrics = True
        for wheel in ['fl', 'fr', 'rl', 'rr']:
            roi = self.torque_extractor.get_roi(wheel)
            card = self.metric_panel.torque_cards.get(wheel)
            if roi and card and card.is_enabled:
                has_metrics = True
                break
        
        if not has_metrics:
            QMessageBox.warning(self, "No Metrics", 
                "No metrics selected for export.\n\n"
                "Please draw ROI boxes for the metrics you want to extract.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            "extracted_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not path:
            return
        
        # Set up progress bar with red styling
        self.progress_bar.setRange(0, self.video_reader.frame_count)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 3px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #e53935;
            }
        """)
        self.progress_bar.show()
        self.status_label.setText("Exporting data...")
        
        # Disable UI during export
        self._pause()
        
        # Create extractors dict
        extractors = {
            'speed': self.speed_extractor,
            'gforce': self.gforce_extractor,
            'kw': self.kw_extractor,
            'throttle': self.throttle_extractor,
            'brake': self.brake_extractor,
            'torque': self.torque_extractor,
        }
        
        # Create worker and thread
        self._export_thread = QThread()
        self._export_worker = ExportWorker(
            self.video_reader,
            extractors,
            path,
            self.metric_panel
        )
        self._export_worker.moveToThread(self._export_thread)
        
        # Connect signals
        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.finished.connect(self._export_thread.quit)
        self._export_worker.finished.connect(self._export_worker.deleteLater)
        self._export_thread.finished.connect(self._export_thread.deleteLater)
        
        # Start export
        self._export_thread.start()
    
    def _on_export_progress(self, current: int, total: int):
        """Update progress bar during export."""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Exporting frame {current}/{total}...")
    
    def _on_export_finished(self, success: bool, message: str):
        """Handle export completion."""
        self.progress_bar.hide()
        # Reset progress bar style
        self.progress_bar.setStyleSheet("")
        
        if success:
            self.status_label.setText("Export complete")
            QMessageBox.information(self, "Export Complete", message)
        else:
            self.status_label.setText("Export failed")
            QMessageBox.warning(self, "Export Failed", message)
    
    # --- About ---
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FB Data Scraper",
            "FB Onboard Video Data Scraper\n\n"
            "Version 0.1.0\n\n"
            "Extract telemetry data from Formula SAE onboard videos "
            "using OCR and computer vision.\n\n"
            "Metrics: Speed, G-Force, Per-wheel Torque"
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        self._pause()
        if self.video_reader:
            self.video_reader.close()
        
        # Cleanup extractors
        self.speed_extractor.cleanup()
        self.gforce_extractor.cleanup()
        
        # Cleanup shared OCR engine
        cleanup_shared_engine()
        
        event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("FB Data Scraper")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("FB")
    
    # Import and show splash screen
    from src.gui.splash import show_splash
    splash = show_splash()
    splash.set_progress(10, "Loading styles...")
    
    # Apply modern stylesheet
    from src.gui.styles import get_full_stylesheet
    app.setStyleSheet(get_full_stylesheet())
    
    splash.set_progress(30, "Initializing window...")
    
    # Create main window
    window = MainWindow()
    
    splash.set_progress(60, "Loading configuration...")
    
    # Brief delay to show splash
    import time
    time.sleep(0.3)
    
    splash.set_progress(100, "Ready!")
    time.sleep(0.2)
    
    # Show window and close splash
    window.show()
    splash.finish(window)
    
    # Auto-initialize EasyOCR after window is shown
    QTimer.singleShot(100, window._initialize_ocr)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
