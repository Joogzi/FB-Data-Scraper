"""Speed extractor using OCR."""

import re
from typing import Optional
import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult, ROI


class SpeedExtractor(BaseExtractor):
    """Extract speed values from video overlay using OCR."""
    
    name = "speed"
    description = "Speed (km/h or mph) via OCR"
    
    def __init__(self, roi: Optional[ROI] = None, use_gpu: bool = True):
        super().__init__(roi)
        self.use_gpu = use_gpu
        self._reader = None
        
        # Preprocessing options
        self.preprocess_grayscale = True
        self.preprocess_threshold = False
        self.threshold_value = 180
        
        # Validation
        self.max_speed = 200  # Maximum valid speed
        self.min_speed = 0
    
    def initialize(self) -> None:
        """Initialize EasyOCR reader."""
        import easyocr
        self._reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        super().initialize()
    
    def cleanup(self) -> None:
        """Release OCR resources."""
        self._reader = None
        super().cleanup()
    
    def preprocess(self, roi_frame: np.ndarray) -> np.ndarray:
        """Preprocess ROI for better OCR accuracy."""
        processed = roi_frame.copy()
        
        if self.preprocess_grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        if self.preprocess_threshold:
            _, processed = cv2.threshold(processed, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        return processed
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract speed value from frame ROI."""
        if self._reader is None:
            return ExtractionResult(value=None, error="OCR not initialized. Call initialize() first.")
        
        # Preprocess
        processed = self.preprocess(frame)
        
        # Run OCR
        try:
            results = self._reader.readtext(processed, detail=0)
            
            if not results:
                return ExtractionResult(value=None, confidence=0.0, error="No text detected")
            
            raw_text = results[0]
            
            # Parse numeric value
            digits = "".join(filter(str.isdigit, raw_text))
            
            if not digits:
                return ExtractionResult(
                    value=None, 
                    confidence=0.0, 
                    raw_output=raw_text,
                    error=f"No digits found in '{raw_text}'"
                )
            
            speed = int(digits)
            
            # Validate range
            if not (self.min_speed <= speed <= self.max_speed):
                return ExtractionResult(
                    value=None,
                    confidence=0.5,
                    raw_output=raw_text,
                    error=f"Speed {speed} out of valid range [{self.min_speed}, {self.max_speed}]"
                )
            
            return ExtractionResult(value=speed, confidence=1.0, raw_output=raw_text)
            
        except Exception as e:
            return ExtractionResult(value=None, error=str(e))
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "use_gpu": self.use_gpu,
            "preprocess_grayscale": self.preprocess_grayscale,
            "preprocess_threshold": self.preprocess_threshold,
            "threshold_value": self.threshold_value,
            "max_speed": self.max_speed,
            "min_speed": self.min_speed,
        })
        return config
