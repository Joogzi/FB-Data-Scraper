"""Video display widget with ROI selection overlay."""

from typing import Callable, Dict, List, Optional, Tuple
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy

import cv2
import numpy as np


class VideoWidget(QWidget):
    """Widget for displaying video frames with ROI overlay and selection."""
    
    roi_selected = pyqtSignal(str, tuple)  # (roi_name, (x1, y1, x2, y2))
    frame_clicked = pyqtSignal(int, int)  # (x, y) in frame coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._frame: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._scale_factor: float = 1.0
        self._offset: QPoint = QPoint(0, 0)
        
        # ROIs to draw
        self._rois: Dict[str, Tuple[int, int, int, int]] = {}
        self._roi_colors: Dict[str, QColor] = {}
        self._roi_values: Dict[str, str] = {}  # Display values
        
        # ROI selection mode
        self._selecting_roi: bool = False
        self._current_roi_name: str = ""
        self._selection_start: Optional[QPoint] = None
        self._selection_rect: Optional[QRect] = None
        
        # Setup
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        
        # Enable focus so parent window receives key events
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        
    def set_frame(self, frame: np.ndarray) -> None:
        """Set the current video frame (BGR format from OpenCV)."""
        self._frame = frame
        self._update_pixmap()
        self.update()
    
    def _update_pixmap(self) -> None:
        """Convert OpenCV frame to QPixmap with scaling."""
        if self._frame is None:
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit widget while maintaining aspect ratio
        widget_size = self.size()
        scaled = qimg.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio, 
                            Qt.TransformationMode.SmoothTransformation)
        
        # Calculate scale factor and offset for coordinate mapping
        self._scale_factor = scaled.width() / w
        self._offset = QPoint(
            (widget_size.width() - scaled.width()) // 2,
            (widget_size.height() - scaled.height()) // 2
        )
        
        self._pixmap = QPixmap.fromImage(scaled)
    
    def _widget_to_frame_coords(self, point: QPoint) -> Tuple[int, int]:
        """Convert widget coordinates to frame coordinates."""
        if self._frame is None:
            return 0, 0
        
        # Remove offset
        x = point.x() - self._offset.x()
        y = point.y() - self._offset.y()
        
        # Apply inverse scale
        frame_x = int(x / self._scale_factor)
        frame_y = int(y / self._scale_factor)
        
        # Clamp to frame bounds
        h, w = self._frame.shape[:2]
        frame_x = max(0, min(w - 1, frame_x))
        frame_y = max(0, min(h - 1, frame_y))
        
        return frame_x, frame_y
    
    def _frame_to_widget_coords(self, x: int, y: int) -> QPoint:
        """Convert frame coordinates to widget coordinates."""
        widget_x = int(x * self._scale_factor) + self._offset.x()
        widget_y = int(y * self._scale_factor) + self._offset.y()
        return QPoint(widget_x, widget_y)
    
    def set_roi(self, name: str, roi: Tuple[int, int, int, int], 
                color: QColor = None, value: str = "") -> None:
        """Set an ROI to display.
        
        Args:
            name: ROI identifier
            roi: (x1, y1, x2, y2) in frame coordinates
            color: Display color
            value: Value text to display
        """
        self._rois[name] = roi
        if color:
            self._roi_colors[name] = color
        if value:
            self._roi_values[name] = value
        self.update()
    
    def clear_roi(self, name: str) -> None:
        """Remove an ROI."""
        self._rois.pop(name, None)
        self._roi_colors.pop(name, None)
        self._roi_values.pop(name, None)
        self.update()
    
    def clear_all_rois(self) -> None:
        """Remove all ROIs."""
        self._rois.clear()
        self._roi_colors.clear()
        self._roi_values.clear()
        self.update()
    
    def update_roi_value(self, name: str, value: str) -> None:
        """Update the displayed value for an ROI."""
        if name in self._rois:
            self._roi_values[name] = value
            self.update()
    
    def start_roi_selection(self, name: str) -> None:
        """Start ROI selection mode for a named ROI."""
        self._selecting_roi = True
        self._current_roi_name = name
        self._selection_start = None
        self._selection_rect = None
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def cancel_roi_selection(self) -> None:
        """Cancel ROI selection mode."""
        self._selecting_roi = False
        self._current_roi_name = ""
        self._selection_start = None
        self._selection_rect = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
    
    def paintEvent(self, event) -> None:
        """Paint the video frame and ROI overlays."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw video frame
        if self._pixmap:
            painter.drawPixmap(self._offset, self._pixmap)
        else:
            # Draw placeholder
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "No video loaded")
        
        # Draw ROIs
        font = QFont("Segoe UI", 10)
        painter.setFont(font)
        
        for name, (x1, y1, x2, y2) in self._rois.items():
            color = self._roi_colors.get(name, QColor(0, 255, 0))
            
            # Convert to widget coordinates
            p1 = self._frame_to_widget_coords(x1, y1)
            p2 = self._frame_to_widget_coords(x2, y2)
            
            # Draw rectangle
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(QRect(p1, p2))
            
            # Draw label
            value = self._roi_values.get(name, "")
            label = f"{name}: {value}" if value else name
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(p1.x(), p1.y() - 5, label)
        
        # Draw selection rectangle
        if self._selection_rect:
            pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._selection_rect)
            
            # Draw instruction
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(10, 30, f"Selecting ROI: {self._current_roi_name}")
        
        painter.end()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for ROI selection."""
        # Set focus to this widget so key events go to parent window
        self.setFocus()
        
        if self._selecting_roi and event.button() == Qt.MouseButton.LeftButton:
            self._selection_start = event.pos()
            self._selection_rect = QRect(self._selection_start, QSize(0, 0))
        else:
            # Emit click in frame coordinates
            if self._frame is not None:
                fx, fy = self._widget_to_frame_coords(event.pos())
                self.frame_clicked.emit(fx, fy)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for ROI selection."""
        if self._selecting_roi and self._selection_start:
            self._selection_rect = QRect(self._selection_start, event.pos()).normalized()
            self.update()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release to complete ROI selection."""
        if self._selecting_roi and event.button() == Qt.MouseButton.LeftButton:
            if self._selection_rect and self._selection_rect.width() > 10 and self._selection_rect.height() > 10:
                # Convert to frame coordinates
                x1, y1 = self._widget_to_frame_coords(self._selection_rect.topLeft())
                x2, y2 = self._widget_to_frame_coords(self._selection_rect.bottomRight())
                
                # Emit the selection
                self.roi_selected.emit(self._current_roi_name, (x1, y1, x2, y2))
                
                # Store the ROI
                self.set_roi(self._current_roi_name, (x1, y1, x2, y2), QColor(0, 255, 0))
            
            self.cancel_roi_selection()
        super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event) -> None:
        """Forward key events to parent (MainWindow) for handling."""
        # Let parent window handle keyboard shortcuts
        if self.parent():
            self.parent().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
    
    def resizeEvent(self, event) -> None:
        """Handle resize to update scaling."""
        self._update_pixmap()
        super().resizeEvent(event)
