"""Torque extractor using color analysis."""

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


# Default color ranges for torque overlays
DEFAULT_GREEN = ColorRange(35, 100, 100, 85, 255, 255)
DEFAULT_RED1 = ColorRange(0, 100, 100, 10, 255, 255)
DEFAULT_RED2 = ColorRange(160, 100, 100, 180, 255, 255)


@dataclass
class TorqueResult:
    """Torque extraction result with detailed color info."""
    value: float  # [-1.0, 1.0] green=positive, red=negative
    green_fraction: float
    red_fraction: float
    valid_fraction: float  # Fraction of ROI with valid colored pixels


class WheelTorqueExtractor(BaseExtractor):
    """Extract torque value from a single wheel ROI using color analysis.
    
    The overlay shows green for positive torque (drive) and red for negative (brake).
    We compute: value = green_fraction - red_fraction in [-1, 1].
    """
    
    name = "wheel_torque"
    description = "Per-wheel torque via color analysis"
    
    def __init__(self, roi: Optional[ROI] = None, wheel_name: str = ""):
        super().__init__(roi)
        self.wheel_name = wheel_name
        
        # Color ranges
        self.green_range = DEFAULT_GREEN
        self.red_range1 = DEFAULT_RED1
        self.red_range2 = DEFAULT_RED2
        
        # Minimum saturation/value to count as valid colored pixel
        self.sat_min = 100
        self.val_min = 100
        
        # Morphology kernel for noise removal
        self.kernel_size = 3
    
    def _create_masks(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create color masks for green, red, and valid pixels."""
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Valid pixel mask (saturated and bright enough)
        valid_mask = ((hsv[:, :, 1] >= self.sat_min) & 
                      (hsv[:, :, 2] >= self.val_min)).astype(np.uint8) * 255
        
        # Green mask
        green_mask = cv2.inRange(hsv, self.green_range.lower, self.green_range.upper)
        
        # Red mask (wraps around H=0, so use two ranges)
        red_mask1 = cv2.inRange(hsv, self.red_range1.lower, self.red_range1.upper)
        red_mask2 = cv2.inRange(hsv, self.red_range2.lower, self.red_range2.upper)
        red_mask = red_mask1 | red_mask2
        
        # Clean up with morphology
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        return green_mask, red_mask, valid_mask
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract torque value from frame (should be ROI already)."""
        if frame.size == 0:
            return ExtractionResult(value=None, error="Empty frame")
        
        green_mask, red_mask, valid_mask = self._create_masks(frame)
        
        # Count pixels
        valid_count = cv2.countNonZero(valid_mask)
        
        if valid_count == 0:
            return ExtractionResult(
                value=TorqueResult(0.0, 0.0, 0.0, 0.0),
                confidence=0.0,
                error="No valid colored pixels found"
            )
        
        # Count green/red within valid area
        green_count = cv2.countNonZero(cv2.bitwise_and(green_mask, valid_mask))
        red_count = cv2.countNonZero(cv2.bitwise_and(red_mask, valid_mask))
        
        # Calculate fractions
        total_pixels = frame.shape[0] * frame.shape[1]
        green_frac = green_count / valid_count
        red_frac = red_count / valid_count
        valid_frac = valid_count / total_pixels
        
        # Compute signed torque value
        value = max(-1.0, min(1.0, green_frac - red_frac))
        
        result = TorqueResult(
            value=value,
            green_fraction=green_frac,
            red_fraction=red_frac,
            valid_fraction=valid_frac
        )
        
        return ExtractionResult(value=result, confidence=valid_frac)
    
    def get_debug_visualization(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Get debug masks for visualization."""
        green_mask, red_mask, valid_mask = self._create_masks(frame)
        return {
            "green": green_mask,
            "red": red_mask,
            "valid": valid_mask
        }
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "wheel_name": self.wheel_name,
            "sat_min": self.sat_min,
            "val_min": self.val_min,
            "kernel_size": self.kernel_size,
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
                # Handle both (y1, y2, x1, x2) and {x1, y1, x2, y2} formats
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
