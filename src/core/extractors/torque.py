"""Torque and bar-based metric extractors using position analysis."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult, ROI


@dataclass
class ColorRange:
    """HSV color range definition."""
    h_min: int
    s_min: int
    v_min: int
    h_max: int
    s_max: int
    v_max: int
    
    @property
    def lower(self) -> np.ndarray:
        return np.array([self.h_min, self.s_min, self.v_min])
    
    @property
    def upper(self) -> np.ndarray:
        return np.array([self.h_max, self.s_max, self.v_max])


# Default color ranges for bar overlays
DEFAULT_GREEN = ColorRange(35, 50, 50, 85, 255, 255)
DEFAULT_RED1 = ColorRange(0, 50, 50, 10, 255, 255)
DEFAULT_RED2 = ColorRange(160, 50, 50, 180, 255, 255)


@dataclass
class BarResult:
    """Bar position extraction result."""
    value: float  # [-1.0, 1.0] - 0 is center/neutral, +1 is top, -1 is bottom
    bar_position: float  # 0.0 = bottom, 1.0 = top
    has_color: bool  # Whether colored bar was detected
    confidence: float


class BarPositionExtractor(BaseExtractor):
    """
    Extract value from a vertical bar based on POSITION, not just color.
    
    The bar works as follows:
    - ROI covers the full range where the bar can move
    - Bar starts in the MIDDLE (neutral = 0)
    - Bar moves UP for positive values (acceleration/torque) - may show green
    - Bar moves DOWN for negative values (brake/negative torque) - may show red
    - When neutral, bar may have no color or be in center position
    
    Detection method:
    1. Find the colored portion of the bar (green or red)
    2. Calculate where the bar edge is relative to center
    3. Return value from -1 to +1 based on position
    """
    
    name = "bar_position"
    description = "Bar value via position analysis"
    
    def __init__(self, roi: Optional[ROI] = None, name: str = ""):
        super().__init__(roi)
        self.bar_name = name
        
        # Color ranges
        self.green_range = DEFAULT_GREEN
        self.red_range1 = DEFAULT_RED1
        self.red_range2 = DEFAULT_RED2
        
        # Detection thresholds
        self.sat_min = 40  # Lower threshold to catch more colors
        self.val_min = 40
        self.min_bar_pixels = 10  # Minimum pixels to consider valid bar
        
        # Morphology
        self.kernel_size = 3
    
    def _detect_bar_position(self, roi_bgr: np.ndarray) -> Tuple[float, bool, str]:
        """
        Detect bar position based on colored region location.
        
        Returns:
            (value, has_color, color_type)
            value: -1.0 to 1.0 (0 = center)
            has_color: whether a colored bar was detected
            color_type: 'green', 'red', or 'none'
        """
        height, width = roi_bgr.shape[:2]
        center_y = height / 2
        
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Create color masks
        green_mask = cv2.inRange(hsv, self.green_range.lower, self.green_range.upper)
        red_mask1 = cv2.inRange(hsv, self.red_range1.lower, self.red_range1.upper)
        red_mask2 = cv2.inRange(hsv, self.red_range2.lower, self.red_range2.upper)
        red_mask = red_mask1 | red_mask2
        
        # Clean up masks
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        green_count = cv2.countNonZero(green_mask)
        red_count = cv2.countNonZero(red_mask)
        
        # Determine which color is dominant
        if green_count >= red_count and green_count >= self.min_bar_pixels:
            mask = green_mask
            color_type = 'green'
        elif red_count > green_count and red_count >= self.min_bar_pixels:
            mask = red_mask
            color_type = 'red'
        else:
            # No significant color detected - neutral position
            return 0.0, False, 'none'
        
        # Find the extent of the colored region
        # For green (positive): bar fills from center UPWARD
        # For red (negative): bar fills from center DOWNWARD
        
        # Get row-wise pixel counts
        row_counts = np.sum(mask > 0, axis=1)
        
        # Find rows with significant bar pixels
        threshold = width * 0.1  # At least 10% of width
        bar_rows = np.where(row_counts >= threshold)[0]
        
        if len(bar_rows) == 0:
            return 0.0, False, 'none'
        
        # Calculate bar extent
        if color_type == 'green':
            # Green bar: extends from center upward
            # The topmost green pixel indicates how far up the bar goes
            top_of_bar = bar_rows[0]  # Smallest y = highest point
            # Calculate how far from center (as fraction of half-height)
            # top_of_bar = 0 means bar at very top = +1.0
            # top_of_bar = center_y means bar at center = 0.0
            value = (center_y - top_of_bar) / center_y
            value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
        else:
            # Red bar: extends from center downward  
            # The bottommost red pixel indicates how far down the bar goes
            bottom_of_bar = bar_rows[-1]  # Largest y = lowest point
            # Calculate how far from center
            # bottom_of_bar = height means bar at very bottom = -1.0
            # bottom_of_bar = center_y means bar at center = 0.0
            value = (bottom_of_bar - center_y) / center_y
            value = -max(0.0, min(1.0, value))  # Clamp to [-1, 0]
        
        return value, True, color_type
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract bar position from frame ROI."""
        if frame.size == 0:
            return ExtractionResult(value=None, error="Empty frame")
        
        value, has_color, color_type = self._detect_bar_position(frame)
        
        # Calculate confidence based on color detection
        confidence = 0.9 if has_color else 0.5
        
        result = BarResult(
            value=value,
            bar_position=(value + 1) / 2,  # Convert to 0-1 range
            has_color=has_color,
            confidence=confidence
        )
        
        return ExtractionResult(value=result, confidence=confidence)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "bar_name": self.bar_name,
            "sat_min": self.sat_min,
            "val_min": self.val_min,
            "min_bar_pixels": self.min_bar_pixels,
        })
        return config


# Keep old TorqueResult for backward compatibility
@dataclass
class TorqueResult:
    """Torque extraction result with detailed info."""
    value: float  # [-1.0, 1.0] - positive=acceleration, negative=brake
    bar_position: float  # 0.0 = bottom, 0.5 = center, 1.0 = top
    has_color: bool
    confidence: float
    
    # Legacy fields for compatibility
    green_fraction: float = 0.0
    red_fraction: float = 0.0
    valid_fraction: float = 0.0


class WheelTorqueExtractor(BarPositionExtractor):
    """Extract torque value from a single wheel ROI using bar position."""
    
    name = "wheel_torque"
    description = "Per-wheel torque via bar position analysis"
    
    def __init__(self, roi: Optional[ROI] = None, wheel_name: str = ""):
        super().__init__(roi, name=wheel_name)
        self.wheel_name = wheel_name
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract torque value with TorqueResult format."""
        if frame.size == 0:
            return ExtractionResult(value=None, error="Empty frame")
        
        value, has_color, color_type = self._detect_bar_position(frame)
        confidence = 0.9 if has_color else 0.5
        
        result = TorqueResult(
            value=value,
            bar_position=(value + 1) / 2,
            has_color=has_color,
            confidence=confidence,
            green_fraction=1.0 if color_type == 'green' else 0.0,
            red_fraction=1.0 if color_type == 'red' else 0.0,
            valid_fraction=confidence
        )
        
        return ExtractionResult(value=result, confidence=confidence)


class PedalExtractor(BarPositionExtractor):
    """
    Extract pedal position (accelerator or brake) from a vertical bar.
    
    For accelerator: bar goes UP from bottom (0% to 100%)
    For brake: bar goes UP from bottom (0% to 100%)
    
    Output is 0.0 to 1.0 (percentage of pedal pressed)
    """
    
    name = "pedal"
    description = "Pedal position via bar analysis"
    
    def __init__(self, roi: Optional[ROI] = None, pedal_type: str = "accelerator"):
        super().__init__(roi, name=pedal_type)
        self.pedal_type = pedal_type  # "accelerator" or "brake"
        
        # Minimum percentage of ROI that must have color to be considered valid
        # This prevents false positives from small noise/artifacts
        self.min_color_ratio = 0.02  # At least 2% of ROI must have color
        
        # Throttle snapping: if above 90% for N frames, snap to 100%
        self._high_throttle_frames = 0
        self._snap_threshold = 0.90  # 90%
        self._snap_frame_count = 3   # Number of frames above threshold to trigger snap
        
        # Adjust color detection based on pedal type
        if pedal_type == "brake":
            # Brake pedal - ONLY look for VIVID red (high saturation, high value)
            self.primary_color = "red"
            # Use higher thresholds - we want bright, saturated red only
            self.red_range1 = ColorRange(0, 120, 100, 12, 255, 255)  # Pure red (high sat/val)
            self.red_range2 = ColorRange(168, 120, 100, 180, 255, 255)  # Wrap-around red
        else:
            # Accelerator - ONLY look for vivid green
            self.primary_color = "green"
            # Override with stricter green detection
            self.green_range = ColorRange(40, 120, 100, 80, 255, 255)
    
    def _detect_pedal_position(self, roi_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Detect pedal position as 0.0 to 1.0.
        
        For pedals, the bar fills from BOTTOM up:
        - 0% = no color (bar at bottom, not visible)
        - 100% = full color (bar fills entire ROI)
        
        Detection method: Scan from TOP down to find where the colored bar starts.
        The position of the top edge of the bar indicates the fill percentage.
        Also validates by checking total color ratio - if position says 90% but
        only 50% of pixels are colored, trust the ratio more (filters out cones/noise).
        
        Returns:
            (position, confidence)
        """
        height, width = roi_bgr.shape[:2]
        
        if height == 0 or width == 0:
            return 0.0, 0.0
        
        total_pixels = height * width
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Create appropriate color mask
        if self.primary_color == "green":
            mask = cv2.inRange(hsv, self.green_range.lower, self.green_range.upper)
        else:
            # For brake - only vivid red
            red_mask1 = cv2.inRange(hsv, self.red_range1.lower, self.red_range1.upper)
            red_mask2 = cv2.inRange(hsv, self.red_range2.lower, self.red_range2.upper)
            mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Light cleanup
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Check if there's enough color in the ROI to be considered valid
        # This prevents false positives from small noise/artifacts
        total_colored_pixels = cv2.countNonZero(mask)
        color_ratio = total_colored_pixels / total_pixels
        
        if color_ratio < self.min_color_ratio:
            # Not enough colored pixels - pedal is at 0%
            return 0.0, 0.8
        
        # Scan each row from TOP (row 0) to BOTTOM (row height-1)
        # to find the first row with significant colored pixels
        min_pixels_per_row = max(2, width * 0.05)  # At least 5% of row width or 2 pixels
        
        top_of_bar = None
        for row_idx in range(height):
            row = mask[row_idx, :]
            colored_pixels = np.sum(row > 0)
            if colored_pixels >= min_pixels_per_row:
                top_of_bar = row_idx
                break
        
        if top_of_bar is None:
            # No colored bar detected - 0% pedal
            return 0.0, 0.7
        
        # Calculate fill percentage based on where the bar starts
        # If top_of_bar = 0, bar fills entire height = 100%
        # If top_of_bar = height-1, bar is just at bottom = ~0%
        # If top_of_bar = height/2, bar fills bottom half = 50%
        
        position_fill = 1.0 - (top_of_bar / height)
        position_fill = max(0.0, min(1.0, position_fill))
        
        # ===== VALIDATION: Compare position-based fill with actual color ratio =====
        # If position says 90% full but only 50% of pixels are colored, something is wrong
        # (probably noise/cones at top of ROI)
        # 
        # For a solid bar, color_ratio should roughly match position_fill
        # Allow some tolerance since bars may not be perfectly solid
        
        # Expected color ratio if bar fills from bottom to position_fill
        # A bar at 50% fill should have ~50% of pixels colored
        # But bars aren't always the full width, so use a factor
        bar_width_factor = 0.7  # Assume bar covers ~70% of ROI width on average
        expected_ratio = position_fill * bar_width_factor
        
        # If actual color is much less than expected, position is wrong (noise at top)
        if position_fill > 0.3:  # Only check for significant fills
            if color_ratio < expected_ratio * 0.5:  # Color is less than half of expected
                # Position is unreliable - use color ratio as estimate instead
                # Scale color_ratio back to fill percentage
                fill_percent = min(1.0, color_ratio / bar_width_factor)
                confidence = 0.7  # Lower confidence since we're using fallback
                return fill_percent, confidence
        
        # Position-based fill looks valid
        fill_percent = position_fill
        
        # Higher confidence if we found a clear bar
        confidence = 0.9
        
        return fill_percent, confidence
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract pedal position from frame ROI."""
        if frame.size == 0:
            return ExtractionResult(value=None, error="Empty frame")
        
        position, confidence = self._detect_pedal_position(frame)
        
        # Throttle snapping to 100% for accelerator pedal
        if self.pedal_type == "accelerator":
            if position >= self._snap_threshold:
                self._high_throttle_frames += 1
                # If above 90% for enough frames, snap to 100%
                if self._high_throttle_frames >= self._snap_frame_count:
                    position = 1.0
            else:
                self._high_throttle_frames = 0
        
        return ExtractionResult(
            value=position,
            confidence=confidence,
            raw_output=f"{position*100:.0f}%"
        )
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "pedal_type": self.pedal_type,
            "primary_color": self.primary_color,
        })
        return config


class FourWheelTorqueExtractor:
    """Manages torque extraction for all four wheels."""
    
    WHEEL_NAMES = ["fl", "fr", "rl", "rr"]
    WHEEL_LABELS = {"fl": "Front Left", "fr": "Front Right", "rl": "Rear Left", "rr": "Rear Right"}
    
    def __init__(self):
        self.extractors: Dict[str, WheelTorqueExtractor] = {
            name: WheelTorqueExtractor(wheel_name=name) for name in self.WHEEL_NAMES
        }
    
    def set_roi(self, wheel: str, roi: ROI) -> None:
        """Set ROI for a specific wheel."""
        if wheel in self.extractors:
            self.extractors[wheel].roi = roi
    
    def get_roi(self, wheel: str) -> Optional[ROI]:
        """Get ROI for a specific wheel."""
        if wheel in self.extractors:
            return self.extractors[wheel].roi
        return None
    
    def extract_all(self, frame: np.ndarray) -> Dict[str, ExtractionResult]:
        """Extract torque values for all wheels."""
        results = {}
        for name, extractor in self.extractors.items():
            if extractor.roi is not None:
                results[name] = extractor.extract_from_roi(frame)
            else:
                results[name] = ExtractionResult(value=None, error="No ROI configured")
        return results
    
    def load_rois(self, rois_dict: Dict[str, Dict]) -> None:
        """Load ROIs from a dictionary."""
        for wheel, roi_data in rois_dict.items():
            if wheel in self.extractors:
                if isinstance(roi_data, (list, tuple)):
                    y1, y2, x1, x2 = roi_data
                    roi = ROI(x1=x1, y1=y1, x2=x2, y2=y2, name=wheel)
                else:
                    roi = ROI.from_dict(roi_data)
                self.set_roi(wheel, roi)
    
    def save_rois(self) -> Dict[str, Dict]:
        """Export ROIs as a dictionary."""
        return {
            name: extractor.roi.to_dict() 
            for name, extractor in self.extractors.items() 
            if extractor.roi is not None
        }
