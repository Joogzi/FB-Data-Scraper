"""Metric configuration panel widget."""

from typing import Callable, Dict, Optional
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QColor


class MetricCard(QGroupBox):
    """Card widget for configuring a single metric."""
    
    select_roi_requested = pyqtSignal(str)  # metric_name
    enabled_changed = pyqtSignal(str, bool)  # metric_name, enabled
    
    def __init__(self, name: str, display_name: str, description: str = "", parent=None):
        super().__init__(display_name, parent)
        self.metric_name = name
        
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Enable checkbox
        top_row = QHBoxLayout()
        self.enable_checkbox = QCheckBox("Enabled")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.stateChanged.connect(self._on_enabled_changed)
        top_row.addWidget(self.enable_checkbox)
        top_row.addStretch()
        
        # Status indicator
        self.status_label = QLabel("No ROI")
        self.status_label.setStyleSheet("color: #ff6b6b;")
        top_row.addWidget(self.status_label)
        layout.addLayout(top_row)
        
        # Description
        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #888; font-style: italic;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
        
        # ROI info
        roi_row = QHBoxLayout()
        self.roi_label = QLabel("ROI: Not set")
        roi_row.addWidget(self.roi_label)
        
        self.select_roi_btn = QPushButton("Select ROI")
        self.select_roi_btn.clicked.connect(lambda: self.select_roi_requested.emit(self.metric_name))
        roi_row.addWidget(self.select_roi_btn)
        layout.addLayout(roi_row)
        
        # Current value display
        value_row = QHBoxLayout()
        value_row.addWidget(QLabel("Current:"))
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        value_row.addWidget(self.value_label)
        value_row.addStretch()
        layout.addLayout(value_row)
        
        # Settings area (can be expanded by subclasses)
        self.settings_widget = QWidget()
        self.settings_layout = QFormLayout(self.settings_widget)
        layout.addWidget(self.settings_widget)
    
    def _on_enabled_changed(self, state: int) -> None:
        enabled = state == Qt.CheckState.Checked.value
        self.enabled_changed.emit(self.metric_name, enabled)
        self._update_appearance()
    
    def _update_appearance(self) -> None:
        enabled = self.enable_checkbox.isChecked()
        self.select_roi_btn.setEnabled(enabled)
        self.setStyleSheet(self.styleSheet().replace(
            "border: 1px solid",
            f"border: 1px solid {'#555' if enabled else '#333'}"
        ))
    
    def set_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Update ROI display."""
        self.roi_label.setText(f"ROI: ({x1}, {y1}) → ({x2}, {y2})")
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #69db7c;")
    
    def set_value(self, value: str) -> None:
        """Update current value display."""
        self.value_label.setText(value)
    
    def set_error(self, error: str) -> None:
        """Show error state."""
        self.status_label.setText(error)
        self.status_label.setStyleSheet("color: #ff6b6b;")
    
    @property
    def is_enabled(self) -> bool:
        return self.enable_checkbox.isChecked()


class SpeedMetricCard(MetricCard):
    """Configuration card for speed metric."""
    
    def __init__(self, parent=None):
        super().__init__(
            "speed", 
            "Speed",
            "Extract speed values using OCR",
            parent
        )
        
        # Add speed-specific settings
        self.max_speed_spin = QSpinBox()
        self.max_speed_spin.setRange(50, 500)
        self.max_speed_spin.setValue(200)
        self.settings_layout.addRow("Max Speed:", self.max_speed_spin)


class GForceMetricCard(MetricCard):
    """Configuration card for G-force metric."""
    
    def __init__(self, parent=None):
        super().__init__(
            "gforce",
            "G-Force", 
            "Extract G-force readings using OCR",
            parent
        )
        
        # Add G-force-specific settings
        self.max_g_spin = QDoubleSpinBox()
        self.max_g_spin.setRange(0.5, 5.0)
        self.max_g_spin.setValue(2.7)
        self.max_g_spin.setSingleStep(0.1)
        self.settings_layout.addRow("Max G:", self.max_g_spin)


class TorqueMetricCard(MetricCard):
    """Configuration card for wheel torque metric."""
    
    def __init__(self, wheel_name: str, display_name: str, parent=None):
        super().__init__(
            f"torque_{wheel_name}",
            display_name,
            "Analyze color overlay for torque visualization",
            parent
        )
        self.wheel_name = wheel_name


class MetricPanel(QScrollArea):
    """Panel containing all metric configuration cards."""
    
    select_roi_requested = pyqtSignal(str)  # metric_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(300)
        
        # Container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        
        # Section: OCR Metrics
        ocr_label = QLabel("OCR Metrics")
        ocr_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #74c0fc;")
        layout.addWidget(ocr_label)
        
        self.speed_card = SpeedMetricCard()
        self.speed_card.select_roi_requested.connect(self.select_roi_requested)
        layout.addWidget(self.speed_card)
        
        self.gforce_card = GForceMetricCard()
        self.gforce_card.select_roi_requested.connect(self.select_roi_requested)
        layout.addWidget(self.gforce_card)
        
        # Section: Torque Metrics
        torque_label = QLabel("Wheel Torque")
        torque_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #74c0fc;")
        layout.addWidget(torque_label)
        
        self.torque_cards: Dict[str, TorqueMetricCard] = {}
        for wheel, name in [("fl", "Front Left"), ("fr", "Front Right"), 
                            ("rl", "Rear Left"), ("rr", "Rear Right")]:
            card = TorqueMetricCard(wheel, name)
            card.select_roi_requested.connect(self.select_roi_requested)
            layout.addWidget(card)
            self.torque_cards[wheel] = card
        
        layout.addStretch()
        self.setWidget(container)
    
    def get_card(self, metric_name: str) -> Optional[MetricCard]:
        """Get the card for a metric by name."""
        if metric_name == "speed":
            return self.speed_card
        elif metric_name == "gforce":
            return self.gforce_card
        elif metric_name.startswith("torque_"):
            wheel = metric_name.replace("torque_", "")
            return self.torque_cards.get(wheel)
        return None
    
    def set_roi(self, metric_name: str, x1: int, y1: int, x2: int, y2: int) -> None:
        """Update ROI for a metric."""
        card = self.get_card(metric_name)
        if card:
            card.set_roi(x1, y1, x2, y2)
    
    def set_value(self, metric_name: str, value: str) -> None:
        """Update displayed value for a metric."""
        card = self.get_card(metric_name)
        if card:
            card.set_value(value)
