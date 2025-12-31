"""G-force extractor using OCR with enhanced preprocessing."""

import re
from typing import Optional
import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult, ROI
from ..ocr_engine import OCREngine, OCRBackend, get_shared_ocr_engine
from ..preprocessor import ImagePreprocessor, PreprocessConfig, PRESETS


class GForceExtractor(BaseExtractor):
    """
    Extract G-force values from video overlay using OCR.
    
    Uses PaddleOCR (with EasyOCR fallback) and advanced preprocessing
    for maximum accuracy across different font styles and overlays.
    """
    
    name = "gforce"
    description = "G-force reading via OCR"
    
    def __init__(self, roi: Optional[ROI] = None, use_gpu: bool = True,
                 preset: str = "racing_hud"):
        super().__init__(roi)
        self.use_gpu = use_gpu
        self.preset = preset
        self._ocr_engine: Optional[OCREngine] = None
        self._preprocessor: Optional[ImagePreprocessor] = None
        
        # Validation - F1 cars can reach 6G+ under braking
        self.max_gforce = 7.0
        self.min_gforce = 0.0
        
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
            if len(roi_frame.shape) == 3:
                return cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            return roi_frame
        
        return self._preprocessor.process(roi_frame)
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract G-force value from frame ROI."""
        if self._ocr_engine is None or not self._ocr_engine.is_initialized:
            return ExtractionResult(value=None, error="OCR not initialized. Call initialize() first.")
        
        # Preprocess
        processed = self.preprocess(frame)
        
        # Run OCR - use specialized numeric reader for decimals
        try:
            value, confidence, raw_text = self._ocr_engine.read_numbers(processed)
            
            if value is None:
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"No numeric value found in '{raw_text}'" if raw_text else "No text detected"
                )
            
            gforce = float(value)
            
            # Check confidence
            if confidence < self.min_confidence:
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"Low confidence ({confidence:.2f}) for '{raw_text}'"
                )
            
            # Validate range
            if not (self.min_gforce <= gforce <= self.max_gforce):
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"G-force {gforce} out of valid range [{self.min_gforce}, {self.max_gforce}]"
                )
            
            return ExtractionResult(value=gforce, confidence=confidence, raw_output=raw_text)
            
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
            "max_gforce": self.max_gforce,
            "min_gforce": self.min_gforce,
            "min_confidence": self.min_confidence,
            "ocr_backend": self._ocr_engine.active_backend if self._ocr_engine else None,
        })
        return config
