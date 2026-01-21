"""
Application styling and theme constants.

Provides a modern, polished dark theme for the FB Data Scraper.
"""

# Color palette
COLORS = {
    # Base colors
    "bg_dark": "#1a1a2e",
    "bg_medium": "#16213e",
    "bg_light": "#0f3460",
    "bg_card": "#1f2940",
    
    # Text colors
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0a0",
    "text_muted": "#6c757d",
    
    # Accent colors
    "accent_primary": "#e94560",
    "accent_secondary": "#0f4c75",
    "accent_highlight": "#3282b8",
    
    # Semantic colors
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    
    # Metric-specific colors
    "speed_color": "#00d4ff",
    "gforce_color": "#ffd700",
    "torque_positive": "#00ff88",
    "torque_negative": "#ff4444",
    
    # Border colors
    "border": "#2d3748",
    "border_light": "#4a5568",
}

# Application stylesheet
STYLESHEET = f"""
/* Main Window */
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QWidget {{
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
}}

/* Menu Bar */
QMenuBar {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 4px;
}}

QMenuBar::item {{
    padding: 6px 12px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['accent_highlight']};
}}

QMenu {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {COLORS['accent_highlight']};
}}

QMenu::separator {{
    height: 1px;
    background: {COLORS['border']};
    margin: 4px 8px;
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLORS['bg_medium']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['text_secondary']};
    padding: 4px;
}}

/* Splitter */
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 2px;
}}

QSplitter::handle:hover {{
    background-color: {COLORS['accent_highlight']};
}}

/* Scroll Area */
QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_light']};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['accent_highlight']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Buttons */
QPushButton {{
    background-color: {COLORS['accent_secondary']};
    color: {COLORS['text_primary']};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS['accent_highlight']};
}}

QPushButton:pressed {{
    background-color: {COLORS['bg_light']};
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_muted']};
}}

/* Primary Button (for important actions) */
QPushButton[primary="true"] {{
    background-color: {COLORS['accent_primary']};
}}

QPushButton[primary="true"]:hover {{
    background-color: #ff5a75;
}}

/* Group Box (Metric Cards) */
QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px;
    padding-top: 24px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: {COLORS['text_primary']};
}}

/* Labels */
QLabel {{
    color: {COLORS['text_primary']};
}}

QLabel[secondary="true"] {{
    color: {COLORS['text_secondary']};
}}

QLabel[value="true"] {{
    font-size: 24px;
    font-weight: bold;
}}

/* Checkboxes */
QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {COLORS['border_light']};
    border-radius: 4px;
    background-color: {COLORS['bg_dark']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['accent_highlight']};
}}

/* Spin Box */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {COLORS['text_primary']};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent_highlight']};
}}

/* Combo Box */
QComboBox {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 12px;
    color: {COLORS['text_primary']};
}}

QComboBox:hover {{
    border-color: {COLORS['accent_highlight']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_highlight']};
}}

/* Progress Bar */
QProgressBar {{
    background-color: {COLORS['bg_dark']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent_primary']};
    border-radius: 4px;
}}

/* Slider (Transport) */
QSlider::groove:horizontal {{
    background-color: {COLORS['bg_dark']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {COLORS['accent_primary']};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: #ff5a75;
}}

QSlider::sub-page:horizontal {{
    background-color: {COLORS['accent_highlight']};
    border-radius: 3px;
}}

/* ToolTip */
QToolTip {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px;
}}

/* Tab Widget (if used) */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    background-color: {COLORS['bg_card']};
}}

QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_secondary']};
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
}}

QTabBar::tab:hover {{
    color: {COLORS['text_primary']};
}}
"""

# Transport controls specific styling
TRANSPORT_STYLE = f"""
/* Transport Control Panel */
QWidget#transport {{
    background-color: {COLORS['bg_medium']};
    border-top: 1px solid {COLORS['border']};
    padding: 8px;
}}

/* Playback buttons */
QPushButton#playBtn {{
    background-color: {COLORS['success']};
    min-width: 48px;
    min-height: 36px;
    font-size: 16px;
}}

QPushButton#playBtn:hover {{
    background-color: #34c759;
}}

QPushButton#pauseBtn {{
    background-color: {COLORS['warning']};
    min-width: 48px;
    min-height: 36px;
}}

QPushButton#stepBtn {{
    background-color: {COLORS['bg_light']};
    min-width: 36px;
}}

/* Time display */
QLabel#timeLabel {{
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 14px;
    color: {COLORS['text_primary']};
    padding: 4px 8px;
    background-color: {COLORS['bg_dark']};
    border-radius: 4px;
}}

/* Frame counter */
QLabel#frameLabel {{
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    color: {COLORS['text_secondary']};
}}
"""

# Video widget styling
VIDEO_WIDGET_STYLE = f"""
QWidget#videoContainer {{
    background-color: {COLORS['bg_dark']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}
"""

# Metric panel styling
METRIC_PANEL_STYLE = f"""
QScrollArea#metricPanel {{
    background-color: {COLORS['bg_medium']};
    border-left: 1px solid {COLORS['border']};
}}

/* Speed metric */
QGroupBox#speedCard {{
    border-left: 4px solid {COLORS['speed_color']};
}}

QLabel#speedValue {{
    color: {COLORS['speed_color']};
    font-size: 28px;
    font-weight: bold;
}}

/* G-Force metric */
QGroupBox#gforceCard {{
    border-left: 4px solid {COLORS['gforce_color']};
}}

QLabel#gforceValue {{
    color: {COLORS['gforce_color']};
    font-size: 28px;
    font-weight: bold;
}}

/* Torque metrics */
QGroupBox[torque="true"] {{
    border-left: 4px solid {COLORS['torque_positive']};
}}
"""


def get_full_stylesheet() -> str:
    """Get the complete application stylesheet."""
    return STYLESHEET + TRANSPORT_STYLE + VIDEO_WIDGET_STYLE + METRIC_PANEL_STYLE


def get_metric_color(metric_name: str) -> str:
    """Get the display color for a specific metric."""
    color_map = {
        "speed": COLORS["speed_color"],
        "gforce": COLORS["gforce_color"],
        "torque_fl": COLORS["torque_positive"],
        "torque_fr": COLORS["torque_positive"],
        "torque_rl": COLORS["torque_positive"],
        "torque_rr": COLORS["torque_positive"],
    }
    return color_map.get(metric_name, COLORS["text_primary"])
