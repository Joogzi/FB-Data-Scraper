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
            
            # PaddleOCR 3.x has a completely new API
            # Try the new API first, then fall back to older versions
            try:
                # New PaddleOCR 3.x API - disable document processing to speed up init
                self._reader = PaddleOCR(
                    lang='en',
                    use_doc_orientation_classify=False,  # Skip document orientation
                    use_doc_unwarping=False,  # Skip document unwarping
                    use_textline_orientation=False,  # Skip textline orientation
                )
                self._initialized = True
                print("PaddleOCR 3.x initialized successfully")
                return True
            except TypeError as e:
                print(f"PaddleOCR 3.x init failed: {e}, trying older API...")
            
            # Try PaddleOCR 2.x API
            try:
                self._reader = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang='en',
                    use_gpu=self.use_gpu,
                    show_log=False,
                    det_db_thresh=0.1,
                    det_db_box_thresh=0.3,
                    det_db_unclip_ratio=1.8,
                )
                self._initialized = True
                print("PaddleOCR 2.x initialized successfully")
                return True
            except TypeError as e:
                print(f"PaddleOCR 2.x init failed: {e}")
                return False
            
        except Exception as e:
            print(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    def read_text(self, image: np.ndarray) -> List[OCRResult]:
        """Read text using PaddleOCR."""
        if not self._initialized or self._reader is None:
            return []
        
        try:
            # PaddleOCR works better with larger images - upscale if too small
            h, w = image.shape[:2]
            scale = 1.0
            if h < 50 or w < 50:
                scale = max(50 / h, 50 / w, 2.0)
                image = cv2.resize(image, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_CUBIC)
            
            # Ensure image is RGB (PaddleOCR expects RGB)
            if len(image.shape) == 2:
                # Grayscale - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # PaddleOCR 3.x uses predict() method, 2.x uses ocr() method
            if hasattr(self._reader, 'predict'):
                # PaddleOCR 3.x API
                result = self._reader.predict(image)
                ocr_results = []
                
                # New API returns a LIST of dicts, each with 'rec_texts' and 'rec_scores'
                # e.g. [{'rec_texts': ['123'], 'rec_scores': [0.99], 'dt_polys': [...]}]
                if result and isinstance(result, list):
                    for page_result in result:
                        if not isinstance(page_result, dict):
                            continue
                        texts = page_result.get('rec_texts', [])
                        scores = page_result.get('rec_scores', [])
                        boxes = page_result.get('dt_polys', []) or page_result.get('rec_polys', [])
                        
                        for i, text in enumerate(texts):
                            if text:
                                confidence = scores[i] if i < len(scores) else 0.5
                                bbox = None
                                if boxes and i < len(boxes) and boxes[i] is not None:
                                    pts = boxes[i]
                                    try:
                                        xs = [p[0] / scale for p in pts]
                                        ys = [p[1] / scale for p in pts]
                                        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                                    except (TypeError, IndexError):
                                        pass
                                
                                ocr_results.append(OCRResult(
                                    text=str(text),
                                    confidence=float(confidence),
                                    bbox=bbox
                                ))
                return ocr_results
            else:
                # PaddleOCR 2.x API
                results = self._reader.ocr(image, cls=self.use_angle_cls)
            
                ocr_results = []
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            bbox_points = line[0]
                            text_info = line[1]
                            
                            if text_info and len(text_info) >= 2:
                                text = str(text_info[0])
                                confidence = float(text_info[1])
                                
                                xs = [p[0] / scale for p in bbox_points]
                                ys = [p[1] / scale for p in bbox_points]
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
            # EasyOCR first - it's MUCH faster for real-time video processing
            # PaddleOCR 3.x is too slow (10+ seconds per frame)
            backends_to_try = [OCRBackend.EASYOCR, OCRBackend.PADDLE]
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
        # Fix common decimal point misreads
        cleaned = cleaned.replace(' . ', '.').replace(' .', '.').replace('. ', '.')
        
        # Find numeric pattern (supports negative and decimal)
        # Try multiple patterns from most specific to least
        patterns = [
            r'-?\d+\.\d+',      # Decimal with digits after (e.g., "1.23")
            r'-?\d+\.?\d*',     # Any number with optional decimal
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    value = float(match.group())
                    return value, confidence, text
                except ValueError:
                    continue
        
        return None, confidence, text
    
    def read_decimal(self, image: np.ndarray, decimal_places: int = 2) -> Tuple[Optional[float], float, str]:
        """
        Specialized method to read decimal values (like G-force, kW).
        
        More aggressive about finding decimal points.
        
        Args:
            image: Input image
            decimal_places: Expected decimal places (for validation)
            
        Returns:
            Tuple of (decimal_value, confidence, raw_text)
        """
        text, confidence = self.read_text_simple(image)
        
        if not text:
            return None, 0.0, ""
        
        # Clean text with more aggressive decimal handling
        cleaned = text.upper().strip()
        
        # Common OCR mistakes
        cleaned = cleaned.replace('O', '0').replace('L', '1').replace('I', '1')
        cleaned = cleaned.replace('S', '5').replace('B', '8').replace('G', '6')
        cleaned = cleaned.replace(',', '.')
        
        # Sometimes OCR reads decimal as space or other chars
        # Try to find patterns like "1 23" -> "1.23" or "1:23" -> "1.23"
        cleaned = re.sub(r'(\d)\s+(\d)', r'\1.\2', cleaned)  # "1 23" -> "1.23"
        cleaned = re.sub(r'(\d)[:\-](\d)', r'\1.\2', cleaned)  # "1:23" or "1-23" -> "1.23"
        
        # Find decimal number pattern
        match = re.search(r'-?\d+\.\d+', cleaned)
        if match:
            try:
                value = float(match.group())
                return value, confidence, text
            except ValueError:
                pass
        
        # Fallback: try to construct decimal if we have multiple digit groups
        digits_only = re.findall(r'\d+', cleaned)
        if len(digits_only) >= 2:
            # Assume first group is integer part, second is decimal
            try:
                value = float(f"{digits_only[0]}.{digits_only[1]}")
                return value, confidence * 0.8, text  # Lower confidence for reconstructed
            except ValueError:
                pass
        elif len(digits_only) == 1 and len(digits_only[0]) >= 2:
            # Single digit group - might be missing decimal
            # For G-force like values, assume format X.XX
            num_str = digits_only[0]
            if len(num_str) == 3:  # e.g., "123" -> "1.23"
                try:
                    value = float(f"{num_str[0]}.{num_str[1:]}")
                    return value, confidence * 0.7, text
                except ValueError:
                    pass
            elif len(num_str) == 2:  # e.g., "12" -> "1.2" or "0.12"
                try:
                    value = float(f"0.{num_str}")
                    return value, confidence * 0.7, text
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
    
    def read_signed_number(self, image: np.ndarray) -> Tuple[Optional[float], float, str]:
        """
        Specialized method to read signed numeric values (can be negative).
        
        Handles cases where:
        - Minus sign is detected as "-" or similar chars
        - OCR might miss the minus sign entirely
        - Values like "-27" should return -27.0
        
        Returns:
            Tuple of (signed_value, confidence, raw_text)
        """
        text, confidence = self.read_text_simple(image)
        
        if not text:
            return None, 0.0, ""
        
        # Clean text
        cleaned = text.upper().strip()
        
        # Common OCR mistakes for digits
        cleaned = cleaned.replace('O', '0').replace('L', '1').replace('I', '1')
        cleaned = cleaned.replace('S', '5').replace('B', '8').replace('G', '6')
        cleaned = cleaned.replace(',', '.')
        
        # Handle minus sign - OCR might read it as various characters
        # Common misreads: '-', '—', '–', '_', '~', 'r', 'v'
        # Check if there's a leading character that could be a minus
        is_negative = False
        
        # Look for minus-like characters at the start
        if cleaned and cleaned[0] in '-—–_~':
            is_negative = True
            cleaned = cleaned[1:].strip()
        elif cleaned.startswith('R') or cleaned.startswith('V'):
            # Sometimes OCR misreads '-' as 'r' or 'v' at start
            # Only treat as negative if followed by digits
            rest = cleaned[1:].strip()
            if rest and rest[0].isdigit():
                is_negative = True
                cleaned = rest
        
        # Also check for explicit "MINUS" or similar
        if cleaned.startswith('MINUS'):
            is_negative = True
            cleaned = cleaned[5:].strip()
        
        # Find numeric pattern
        patterns = [
            r'\d+\.\d+',      # Decimal with digits after
            r'\d+',            # Integer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    value = float(match.group())
                    if is_negative:
                        value = -value
                    return value, confidence, text
                except ValueError:
                    continue
        
        return None, confidence, text
    
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
_shared_backend: Optional[OCRBackend] = None
_shared_use_gpu: bool = True


def get_shared_ocr_engine(backend: OCRBackend = OCRBackend.AUTO, 
                          use_gpu: bool = True) -> OCREngine:
    """
    Get or create a shared OCR engine instance.
    
    This is useful to avoid initializing multiple OCR engines,
    which would waste memory and time.
    
    If backend/gpu settings differ from current engine, recreates it.
    """
    global _shared_engine, _shared_backend, _shared_use_gpu
    
    # Recreate if settings changed or not initialized
    if (_shared_engine is None or 
        _shared_backend != backend or 
        _shared_use_gpu != use_gpu):
        
        if _shared_engine is not None:
            _shared_engine.cleanup()
        
        _shared_engine = OCREngine(backend=backend, use_gpu=use_gpu)
        _shared_backend = backend
        _shared_use_gpu = use_gpu
    
    return _shared_engine


def cleanup_shared_engine() -> None:
    """Cleanup the shared OCR engine."""
    global _shared_engine, _shared_backend, _shared_use_gpu
    if _shared_engine:
        _shared_engine.cleanup()
        _shared_engine = None
        _shared_backend = None
        _shared_use_gpu = True
