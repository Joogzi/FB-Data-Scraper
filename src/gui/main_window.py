"""Main application window for FSAE Data Extractor."""

import sys
import os
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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
from src.core.extractors.torque import FourWheelTorqueExtractor, TorqueResult
from src.config.settings import ConfigManager, MetricConfig
from src.gui.widgets.video_widget import VideoWidget
from src.gui.widgets.metric_panel import MetricPanel
from src.gui.widgets.transport import TransportControls


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("FSAE Data Extractor")
        self.setMinimumSize(1200, 800)
        
        # Core components
        self.video_reader: Optional[VideoReader] = None
        self.config_manager = ConfigManager()
        
        # Extractors
        self.speed_extractor = SpeedExtractor()
        self.gforce_extractor = GForceExtractor()
        self.torque_extractor = FourWheelTorqueExtractor()
        
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
        
        init_ocr_action = QAction("Initialize &OCR", self)
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
        elif key == Qt.Key.Key_Right:
            self._step(1 if not event.modifiers() else 10)
        elif key == Qt.Key.Key_Left:
            self._step(-1 if not event.modifiers() else -10)
        elif key == Qt.Key.Key_Up:
            self._step(10)
        elif key == Qt.Key.Key_Down:
            self._step(-10)
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
        # Speed
        if self.speed_extractor.roi and self.metric_panel.speed_card.is_enabled:
            result = self.speed_extractor.extract_from_roi(frame)
            if result.is_valid:
                self.metric_panel.set_value("speed", f"{result.value} km/h")
                self.video_widget.update_roi_value("speed", f"{result.value}")
            else:
                self.metric_panel.set_value("speed", "--")
        
        # G-force
        if self.gforce_extractor.roi and self.metric_panel.gforce_card.is_enabled:
            result = self.gforce_extractor.extract_from_roi(frame)
            if result.is_valid:
                self.metric_panel.set_value("gforce", f"{result.value:.2f} G")
                self.video_widget.update_roi_value("gforce", f"{result.value:.2f}G")
            else:
                self.metric_panel.set_value("gforce", "--")
        
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
        elif metric_name.startswith("torque_"):
            wheel = metric_name.replace("torque_", "")
            self.torque_extractor.set_roi(wheel, roi)
        
        # Update UI
        self.metric_panel.set_roi(metric_name, x1, y1, x2, y2)
        
        # Set display color based on metric type
        if metric_name == "speed":
            color = QColor(0, 200, 255)
        elif metric_name == "gforce":
            color = QColor(255, 200, 0)
        else:
            color = QColor(0, 255, 0)
        
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
        """Initialize OCR engines."""
        self.status_label.setText("Initializing OCR...")
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        QApplication.processEvents()
        
        try:
            self.speed_extractor.initialize()
            self.gforce_extractor.initialize()
            self.status_label.setText("OCR initialized")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize OCR: {e}")
            self.status_label.setText("OCR initialization failed")
        finally:
            self.progress_bar.hide()
    
    # --- Export ---
    
    def _export_data(self):
        """Export extracted data to CSV."""
        if self.video_reader is None:
            QMessageBox.warning(self, "No Video", "Please open a video first.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            "extracted_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not path:
            return
        
        # TODO: Implement batch extraction and export
        QMessageBox.information(self, "Export", 
            "Batch export will process all frames and save to CSV.\n"
            "This feature is coming soon!")
    
    # --- About ---
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FSAE Data Extractor",
            "FSAE Onboard Video Data Extractor\n\n"
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
        
        event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle("Fusion")
    
    # Dark palette
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
