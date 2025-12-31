"""Speed extractor using OCR with enhanced preprocessing."""

import re
from typing import Optional
import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult, ROI
from ..ocr_engine import OCREngine, OCRBackend, get_shared_ocr_engine
from ..preprocessor import ImagePreprocessor, PreprocessConfig, PRESETS


class SpeedExtractor(BaseExtractor):
    """
    Extract speed values from video overlay using OCR.
    
    Uses PaddleOCR (with EasyOCR fallback) and advanced preprocessing
    for maximum accuracy across different font styles and overlays.
    """
    
    name = "speed"
    description = "Speed (km/h or mph) via OCR"
    
    def __init__(self, roi: Optional[ROI] = None, use_gpu: bool = True, 
                 preset: str = "racing_hud"):
        super().__init__(roi)
        self.use_gpu = use_gpu
        self.preset = preset
        self._ocr_engine: Optional[OCREngine] = None
        self._preprocessor: Optional[ImagePreprocessor] = None
        
        # Validation
        self.max_speed = 400  # F1 cars can reach 370+ km/h
        self.min_speed = 0
        
        # Confidence threshold
        self.min_confidence = 0.5
        
        # Use shared OCR engine to save memory
        self.use_shared_engine = True
    
    def initialize(self) -> None:
        """Initialize OCR engine and preprocessor."""
        if self.use_shared_engine:
            self._ocr_engine = get_shared_ocr_engine(
                backend=OCRBackend.AUTO, 
                use_gpu=self.use_gpu
            )
        else:
            self._ocr_engine = OCREngine(
                backend=OCRBackend.AUTO,
                use_gpu=self.use_gpu
            )
        
        if not self._ocr_engine.is_initialized:
            self._ocr_engine.initialize()
        
        # Setup preprocessor
        config = PRESETS.get(self.preset, PRESETS["racing_hud"])
        self._preprocessor = ImagePreprocessor(config)
        
        super().initialize()
    
    def cleanup(self) -> None:
        """Release OCR resources."""
        if not self.use_shared_engine and self._ocr_engine:
            self._ocr_engine.cleanup()
        self._ocr_engine = None
        self._preprocessor = None
        super().cleanup()
    
    def preprocess(self, roi_frame: np.ndarray) -> np.ndarray:
        """Preprocess ROI for better OCR accuracy."""
        if self._preprocessor is None:
            # Fallback to basic preprocessing
            if len(roi_frame.shape) == 3:
                return cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            return roi_frame
        
        return self._preprocessor.process(roi_frame)
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract speed value from frame ROI."""
        if self._ocr_engine is None or not self._ocr_engine.is_initialized:
            return ExtractionResult(value=None, error="OCR not initialized. Call initialize() first.")
        
        # Preprocess
        processed = self.preprocess(frame)
        
        # Run OCR - use specialized integer reader
        try:
            value, confidence, raw_text = self._ocr_engine.read_integer(processed)
            
            if value is None:
                return ExtractionResult(
                    value=None, 
                    confidence=confidence, 
                    raw_output=raw_text,
                    error=f"No digits found in '{raw_text}'" if raw_text else "No text detected"
                )
            
            # Check confidence
            if confidence < self.min_confidence:
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"Low confidence ({confidence:.2f}) for '{raw_text}'"
                )
            
            # Validate range
            if not (self.min_speed <= value <= self.max_speed):
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"Speed {value} out of valid range [{self.min_speed}, {self.max_speed}]"
                )
            
            return ExtractionResult(value=value, confidence=confidence, raw_output=raw_text)
            
        except Exception as e:
            return ExtractionResult(value=None, error=str(e))
    
    def set_preset(self, preset: str) -> None:
        """Change preprocessing preset."""
        self.preset = preset
        if preset in PRESETS:
            self._preprocessor = ImagePreprocessor(PRESETS[preset])
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "use_gpu": self.use_gpu,
            "preset": self.preset,
            "max_speed": self.max_speed,
            "min_speed": self.min_speed,
            "min_confidence": self.min_confidence,
            "ocr_backend": self._ocr_engine.active_backend if self._ocr_engine else None,
        })
        return config
