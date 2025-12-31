"""G-force extractor using OCR."""

import re
from typing import Optional
import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult, ROI


class GForceExtractor(BaseExtractor):
    """Extract G-force values from video overlay using OCR."""
    
    name = "gforce"
    description = "G-force reading via OCR"
    
    def __init__(self, roi: Optional[ROI] = None, use_gpu: bool = True):
        super().__init__(roi)
        self.use_gpu = use_gpu
        self._reader = None
        
        # Preprocessing options
        self.preprocess_grayscale = True
        
        # Validation - FSAE cars typically don't exceed 2.5G
        self.max_gforce = 2.7
        self.min_gforce = 0.0
    
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
        
        return processed
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract G-force value from frame ROI."""
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
            
            # Parse floating point value (e.g., "1.23", "0.45")
            match = re.search(r"\d+(\.\d+)?", raw_text)
            
            if not match:
                return ExtractionResult(
                    value=None,
                    confidence=0.0,
                    raw_output=raw_text,
                    error=f"No numeric value found in '{raw_text}'"
                )
            
            gforce = float(match.group())
            
            # Validate range
            if not (self.min_gforce <= gforce <= self.max_gforce):
                return ExtractionResult(
                    value=None,
                    confidence=0.5,
                    raw_output=raw_text,
                    error=f"G-force {gforce} out of valid range [{self.min_gforce}, {self.max_gforce}]"
                )
            
            return ExtractionResult(value=gforce, confidence=1.0, raw_output=raw_text)
            
        except Exception as e:
            return ExtractionResult(value=None, error=str(e))
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "use_gpu": self.use_gpu,
            "preprocess_grayscale": self.preprocess_grayscale,
            "max_gforce": self.max_gforce,
            "min_gforce": self.min_gforce,
        })
        return config
