"""
Splash screen for application startup.

Shows a professional loading screen while OCR and other
heavy components are being initialized.
"""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QLinearGradient, QPen
from PyQt6.QtWidgets import QSplashScreen, QApplication


class SplashScreen(QSplashScreen):
    """Custom splash screen with progress indication."""
    
    def __init__(self):
        # Create a custom pixmap for the splash screen
        pixmap = self._create_splash_pixmap()
        super().__init__(pixmap)
        
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        
        self._message = "Starting..."
        self._progress = 0
    
    def _create_splash_pixmap(self) -> QPixmap:
        """Create the splash screen background."""
        width, height = 500, 300
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor("#1a1a2e"))
        gradient.setColorAt(1, QColor("#16213e"))
        
        # Rounded rectangle background
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor("#e94560"), 2))
        painter.drawRoundedRect(1, 1, width - 2, height - 2, 15, 15)
        
        # App title
        title_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(0, 60, width, 50, Qt.AlignmentFlag.AlignCenter, "FB Data Scraper")
        
        # Subtitle
        subtitle_font = QFont("Segoe UI", 12)
        painter.setFont(subtitle_font)
        painter.setPen(QColor("#a0a0a0"))
        painter.drawText(0, 110, width, 30, Qt.AlignmentFlag.AlignCenter, 
                        "Video Telemetry Analysis Tool")
        
        # Accent line
        painter.setPen(QPen(QColor("#e94560"), 3))
        painter.drawLine(150, 150, 350, 150)
        
        # Version
        version_font = QFont("Segoe UI", 10)
        painter.setFont(version_font)
        painter.setPen(QColor("#6c757d"))
        painter.drawText(0, height - 40, width, 20, Qt.AlignmentFlag.AlignCenter, 
                        "Version 1.0.0")
        
        painter.end()
        return pixmap
    
    def set_progress(self, value: int, message: str = "") -> None:
        """Update progress and message."""
        self._progress = min(100, max(0, value))
        if message:
            self._message = message
        self.showMessage(
            f"  {self._message}",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            QColor("#a0a0a0")
        )
        QApplication.processEvents()
    
    def drawContents(self, painter: QPainter) -> None:
        """Draw splash contents including progress bar."""
        super().drawContents(painter)
        
        # Draw progress bar
        bar_x, bar_y = 50, 200
        bar_width, bar_height = 400, 6
        
        # Background
        painter.setBrush(QColor("#0f3460"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 3, 3)
        
        # Progress fill
        if self._progress > 0:
            fill_width = int(bar_width * self._progress / 100)
            gradient = QLinearGradient(bar_x, 0, bar_x + fill_width, 0)
            gradient.setColorAt(0, QColor("#e94560"))
            gradient.setColorAt(1, QColor("#ff6b8a"))
            painter.setBrush(gradient)
            painter.drawRoundedRect(bar_x, bar_y, fill_width, bar_height, 3, 3)


def show_splash() -> SplashScreen:
    """Create and show splash screen."""
    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()
    return splash
