"""
OCR Engine module with multiple backend support.

Supports:
- PaddleOCR (recommended - best accuracy for varied fonts)
- EasyOCR (fallback)

PaddleOCR provides significantly better accuracy on:
- Different font styles and weights
- Numeric text (critical for telemetry)
- Various video overlay styles
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class OCRBackend(Enum):
    """Available OCR backends."""
    PADDLE = "paddle"
    EASYOCR = "easyocr"
    AUTO = "auto"  # Try PaddleOCR first, fallback to EasyOCR


@dataclass
class OCRResult:
    """Result from OCR engine."""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    
    @property
    def is_valid(self) -> bool:
        return len(self.text.strip()) > 0 and self.confidence > 0.3


class BaseOCREngine(ABC):
    """Base class for OCR engines."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine. Returns True on success."""
        pass
    
    @abstractmethod
    def read_text(self, image: np.ndarray) -> List[OCRResult]:
        """Read text from image. Returns list of OCRResult."""
        pass
    
    def cleanup(self) -> None:
        """Release resources."""
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR-based engine - best accuracy for varied fonts."""
    
    def __init__(self, use_gpu: bool = True, use_angle_cls: bool = False):
        super().__init__(use_gpu)
        self.use_angle_cls = use_angle_cls
        self._reader = None
    
    def initialize(self) -> bool:
        """Initialize PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            
            self._reader = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang='en',
                use_gpu=self.use_gpu,
                show_log=False,  # Suppress verbose logging
                # Optimize for single-line numeric text
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=1,
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    def read_text(self, image: np.ndarray) -> List[OCRResult]:
        """Read text using PaddleOCR."""
        if not self._initialized or self._reader is None:
            return []
        
        try:
            # PaddleOCR expects RGB, but works with BGR too
            results = self._reader.ocr(image, cls=self.use_angle_cls)
            
            ocr_results = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = line[1]  # (text, confidence)
                        
                        if text_info and len(text_info) >= 2:
                            text = str(text_info[0])
                            confidence = float(text_info[1])
                            
                            # Convert polygon to bbox
                            xs = [p[0] for p in bbox_points]
                            ys = [p[1] for p in bbox_points]
                            bbox = (int(min(xs)), int(min(ys)), 
                                   int(max(xs)), int(max(ys)))
                            
                            ocr_results.append(OCRResult(
                                text=text,
                                confidence=confidence,
                                bbox=bbox
                            ))
            
            return ocr_results
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return []
    
    def cleanup(self) -> None:
        self._reader = None
        super().cleanup()


class EasyOCREngine(BaseOCREngine):
    """EasyOCR-based engine - fallback option."""
    
    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self._reader = None
    
    def initialize(self) -> bool:
        """Initialize EasyOCR."""
        try:
            import easyocr
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            return False
    
    def read_text(self, image: np.ndarray) -> List[OCRResult]:
        """Read text using EasyOCR."""
        if not self._initialized or self._reader is None:
            return []
        
        try:
            # EasyOCR returns: [[bbox, text, confidence], ...]
            results = self._reader.readtext(image)
            
            ocr_results = []
            for result in results:
                bbox_points = result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = str(result[1])
                confidence = float(result[2])
                
                # Convert polygon to bbox
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                bbox = (int(min(xs)), int(min(ys)), 
                       int(max(xs)), int(max(ys)))
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                ))
            
            return ocr_results
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return []
    
    def cleanup(self) -> None:
        self._reader = None
        super().cleanup()


class OCREngine:
    """
    Unified OCR engine with automatic backend selection.
    
    Usage:
        engine = OCREngine(backend=OCRBackend.AUTO)
        engine.initialize()
        results = engine.read_text(image)
        text = engine.read_text_simple(image)  # Just get the text
    """
    
    def __init__(self, backend: OCRBackend = OCRBackend.AUTO, use_gpu: bool = True):
        self.backend_type = backend
        self.use_gpu = use_gpu
        self._engine: Optional[BaseOCREngine] = None
        self._active_backend: Optional[OCRBackend] = None
    
    def initialize(self) -> bool:
        """Initialize OCR engine with automatic fallback."""
        backends_to_try = []
        
        if self.backend_type == OCRBackend.AUTO:
            backends_to_try = [OCRBackend.PADDLE, OCRBackend.EASYOCR]
        else:
            backends_to_try = [self.backend_type]
        
        for backend in backends_to_try:
            engine = self._create_engine(backend)
            if engine and engine.initialize():
                self._engine = engine
                self._active_backend = backend
                print(f"OCR initialized with {backend.value}")
                return True
        
        print("Failed to initialize any OCR backend")
        return False
    
    def _create_engine(self, backend: OCRBackend) -> Optional[BaseOCREngine]:
        """Create engine instance for specified backend."""
        if backend == OCRBackend.PADDLE:
            return PaddleOCREngine(use_gpu=self.use_gpu)
        elif backend == OCRBackend.EASYOCR:
            return EasyOCREngine(use_gpu=self.use_gpu)
        return None
    
    def read_text(self, image: np.ndarray) -> List[OCRResult]:
        """Read all text from image."""
        if self._engine is None:
            return []
        return self._engine.read_text(image)
    
    def read_text_simple(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Read text and return just the combined text and average confidence.
        
        Returns:
            Tuple of (text, confidence)
        """
        results = self.read_text(image)
        if not results:
            return "", 0.0
        
        # Combine all text, sort by position (top to bottom, left to right)
        results.sort(key=lambda r: (r.bbox[1] if r.bbox else 0, r.bbox[0] if r.bbox else 0))
        
        combined_text = " ".join(r.text for r in results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return combined_text, avg_confidence
    
    def read_numbers(self, image: np.ndarray) -> Tuple[Optional[float], float, str]:
        """
        Specialized method to read numeric values.
        
        Returns:
            Tuple of (numeric_value, confidence, raw_text)
        """
        text, confidence = self.read_text_simple(image)
        
        if not text:
            return None, 0.0, ""
        
        # Extract numeric value (handles decimals, negatives)
        # Common OCR mistakes: O->0, l->1, I->1, S->5, B->8
        cleaned = text.upper()
        cleaned = cleaned.replace('O', '0').replace('L', '1').replace('I', '1')
        cleaned = cleaned.replace('S', '5').replace('B', '8').replace('G', '6')
        cleaned = cleaned.replace(',', '.')  # Handle European decimal notation
        
        # Find numeric pattern (supports negative and decimal)
        match = re.search(r'-?\d+\.?\d*', cleaned)
        
        if match:
            try:
                value = float(match.group())
                return value, confidence, text
            except ValueError:
                pass
        
        return None, confidence, text
    
    def read_integer(self, image: np.ndarray) -> Tuple[Optional[int], float, str]:
        """
        Specialized method to read integer values (like speed).
        
        Returns:
            Tuple of (integer_value, confidence, raw_text)
        """
        value, confidence, raw_text = self.read_numbers(image)
        
        if value is not None:
            return int(round(value)), confidence, raw_text
        
        return None, confidence, raw_text
    
    @property
    def is_initialized(self) -> bool:
        return self._engine is not None and self._engine.is_initialized
    
    @property
    def active_backend(self) -> Optional[str]:
        return self._active_backend.value if self._active_backend else None
    
    def cleanup(self) -> None:
        if self._engine:
            self._engine.cleanup()
        self._engine = None
        self._active_backend = None


# Singleton instance for shared OCR engine
_shared_engine: Optional[OCREngine] = None


def get_shared_ocr_engine(backend: OCRBackend = OCRBackend.AUTO, 
                          use_gpu: bool = True) -> OCREngine:
    """
    Get or create a shared OCR engine instance.
    
    This is useful to avoid initializing multiple OCR engines,
    which would waste memory and time.
    """
    global _shared_engine
    
    if _shared_engine is None:
        _shared_engine = OCREngine(backend=backend, use_gpu=use_gpu)
    
    return _shared_engine


def cleanup_shared_engine() -> None:
    """Cleanup the shared OCR engine."""
    global _shared_engine
    if _shared_engine:
        _shared_engine.cleanup()
        _shared_engine = None
