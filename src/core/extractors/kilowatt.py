"""Kilowatt (power) extractor using OCR with decimal support."""

from typing import Optional
import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult, ROI
from ..ocr_engine import OCREngine, OCRBackend, get_shared_ocr_engine
from ..preprocessor import ImagePreprocessor, PreprocessConfig, PRESETS


class KilowattExtractor(BaseExtractor):
    """
    Extract kilowatt (power) values from video overlay using OCR.
    
    Expects values in x.xx format (e.g., "1.23", "0.50", "2.00").
    Uses PaddleOCR (with EasyOCR fallback) for best accuracy.
    Includes temporal smoothing to detect and correct missed decimal points.
    Can use pedal data to inform sign correction.
    """
    
    name = "kilowatt"
    description = "Power (kW) via OCR - x.xx format"
    
    def __init__(self, roi: Optional[ROI] = None, use_gpu: bool = True,
                 preset: str = "racing_hud"):
        super().__init__(roi)
        self.use_gpu = use_gpu
        self.preset = preset
        self._ocr_engine: Optional[OCREngine] = None
        self._preprocessor: Optional[ImagePreprocessor] = None
        
        # Validation - typical EV/hybrid power range (can be negative for regen)
        self.max_kw = 120.0   # Maximum valid kW (user configurable)
        self.min_kw = -120.0  # Negative for regenerative braking
        
        # Confidence threshold
        self.min_confidence = 0.4
        
        # Use shared OCR engine
        self.use_shared_engine = True
        
        # OCR backend preference (can be overridden)
        self.ocr_backend = OCRBackend.AUTO
        
        # Temporal smoothing - track previous value to detect OCR errors
        self._last_valid_value: Optional[float] = None
        self._max_change_threshold = 50.0  # Max reasonable change between frames
        
        # Context for sign correction (set externally before extract)
        self.brake_pct: float = 0.0     # 0-100
        self.speed: float = 0.0         # km/h
    
    def initialize(self) -> None:
        """Initialize OCR engine and preprocessor."""
        if self.use_shared_engine:
            self._ocr_engine = get_shared_ocr_engine(
                backend=self.ocr_backend, 
                use_gpu=self.use_gpu
            )
        else:
            self._ocr_engine = OCREngine(
                backend=self.ocr_backend,
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
        """Extract kilowatt value from frame ROI - simple OCR with jump prevention."""
        if self._ocr_engine is None or not self._ocr_engine.is_initialized:
            return ExtractionResult(value=None, error="OCR not initialized. Call initialize() first.")
        
        # Preprocess
        processed = self.preprocess(frame)
        
        # Run OCR - use numeric reader with signed number support
        try:
            value, confidence, raw_text = self._ocr_engine.read_signed_number(processed)
            
            if value is None:
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"No numeric value found in '{raw_text}'" if raw_text else "No text detected"
                )
            
            kw = float(value)
            
            # Check confidence
            if confidence < self.min_confidence:
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"Low confidence ({confidence:.2f}) for '{raw_text}'"
                )
            
            # Only correction: if value is WAY outside range (like 500), divide by 10
            if abs(kw) > abs(self.max_kw):
                corrected = kw / 10.0
                if abs(corrected) <= abs(self.max_kw):
                    kw = corrected
            
            # ===== SIGN CORRECTION =====
            # If speed > 10 km/h AND brake is active, kW MUST be negative (regen braking)
            # OCR sometimes misses the minus sign
            if self.speed > 10 and self.brake_pct > 5:
                if kw > 0:
                    kw = -kw  # Flip to negative
            
            # ===== JUMP PREVENTION =====
            # Prevent impossible jumps like -50 to 50 in one frame
            if self._last_valid_value is not None:
                change = abs(kw - self._last_valid_value)
                
                # If change is too big, reject this value
                if change > self._max_change_threshold:
                    return ExtractionResult(
                        value=None,
                        confidence=confidence,
                        raw_output=raw_text,
                        error=f"Value {kw:.1f} jumped too far from {self._last_valid_value:.1f} (change: {change:.1f})"
                    )
            
            # Final range validation
            if not (self.min_kw <= kw <= self.max_kw):
                return ExtractionResult(
                    value=None,
                    confidence=confidence,
                    raw_output=raw_text,
                    error=f"Power {kw} out of valid range [{self.min_kw}, {self.max_kw}]"
                )
            
            # Update last valid value for next frame
            self._last_valid_value = kw
            
            return ExtractionResult(value=kw, confidence=confidence, raw_output=raw_text)
            
        except Exception as e:
            return ExtractionResult(value=None, error=str(e))
    
    def reset_tracking(self):
        """Reset the last valid value tracking (call when seeking to new position)."""
        self._last_valid_value = None
    
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
            "max_kw": self.max_kw,
            "min_kw": self.min_kw,
            "min_confidence": self.min_confidence,
            "ocr_backend": self._ocr_engine.active_backend if self._ocr_engine else None,
        })
        return config
