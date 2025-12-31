"""Video playback transport controls widget."""

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
    QSlider, QLabel, QSpinBox, QStyle
)


class TransportControls(QWidget):
    """Video playback transport controls (play, pause, seek, etc.)."""
    
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    seek_requested = pyqtSignal(int)  # frame index
    step_forward = pyqtSignal(int)  # step size
    step_backward = pyqtSignal(int)  # step size
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._frame_count = 0
        self._fps = 30.0
        self._current_frame = 0
        self._is_playing = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Timeline slider
        slider_row = QHBoxLayout()
        
        self.time_label = QLabel("00:00.00")
        self.time_label.setMinimumWidth(70)
        slider_row.addWidget(self.time_label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider)
        
        self.duration_label = QLabel("00:00.00")
        self.duration_label.setMinimumWidth(70)
        slider_row.addWidget(self.duration_label)
        
        layout.addLayout(slider_row)
        
        # Controls row
        controls_row = QHBoxLayout()
        controls_row.setSpacing(5)
        
        # Step backward
        self.step_back_btn = QPushButton("◀◀")
        self.step_back_btn.setFixedWidth(40)
        self.step_back_btn.setToolTip("Step backward (Left Arrow)")
        self.step_back_btn.clicked.connect(lambda: self.step_backward.emit(1))
        controls_row.addWidget(self.step_back_btn)
        
        # Previous frame
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.setToolTip("Previous frame")
        self.prev_btn.clicked.connect(lambda: self.step_backward.emit(1))
        controls_row.addWidget(self.prev_btn)
        
        # Play/Pause
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setFixedWidth(50)
        self.play_pause_btn.setToolTip("Play/Pause (Space)")
        self.play_pause_btn.clicked.connect(self._on_play_pause)
        controls_row.addWidget(self.play_pause_btn)
        
        # Next frame
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(40)
        self.next_btn.setToolTip("Next frame")
        self.next_btn.clicked.connect(lambda: self.step_forward.emit(1))
        controls_row.addWidget(self.next_btn)
        
        # Step forward
        self.step_fwd_btn = QPushButton("▶▶")
        self.step_fwd_btn.setFixedWidth(40)
        self.step_fwd_btn.setToolTip("Step forward (Right Arrow)")
        self.step_fwd_btn.clicked.connect(lambda: self.step_forward.emit(10))
        controls_row.addWidget(self.step_fwd_btn)
        
        controls_row.addStretch()
        
        # Frame info
        controls_row.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        controls_row.addWidget(self.frame_spin)
        
        self.total_frames_label = QLabel("/ 0")
        controls_row.addWidget(self.total_frames_label)
        
        controls_row.addWidget(QLabel(" | FPS:"))
        self.fps_label = QLabel("30.0")
        controls_row.addWidget(self.fps_label)
        
        layout.addLayout(controls_row)
    
    def set_video_info(self, frame_count: int, fps: float):
        """Configure for a video."""
        self._frame_count = frame_count
        self._fps = fps
        
        self.slider.setRange(0, frame_count - 1 if frame_count > 0 else 0)
        self.frame_spin.setRange(0, frame_count - 1 if frame_count > 0 else 0)
        self.total_frames_label.setText(f"/ {frame_count}")
        self.fps_label.setText(f"{fps:.1f}")
        
        duration = frame_count / fps if fps > 0 else 0
        self.duration_label.setText(self._format_time(duration))
    
    def set_current_frame(self, frame: int):
        """Update current frame display."""
        self._current_frame = frame
        
        # Block signals to avoid feedback loops
        self.slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        
        self.slider.setValue(frame)
        self.frame_spin.setValue(frame)
        
        self.slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        
        time_sec = frame / self._fps if self._fps > 0 else 0
        self.time_label.setText(self._format_time(time_sec))
    
    def set_playing(self, playing: bool):
        """Update play/pause button state."""
        self._is_playing = playing
        self.play_pause_btn.setText("⏸" if playing else "▶")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS.cc"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
    
    def _on_slider_changed(self, value: int):
        self.seek_requested.emit(value)
    
    def _on_frame_spin_changed(self, value: int):
        self.seek_requested.emit(value)
    
    def _on_play_pause(self):
        if self._is_playing:
            self.pause_clicked.emit()
        else:
            self.play_clicked.emit()
