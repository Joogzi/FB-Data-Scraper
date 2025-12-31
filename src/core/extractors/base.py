"""Base extractor class for all metric extractors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np


@dataclass
class ROI:
    """Region of Interest definition."""
    x1: int
    y1: int
    x2: int
    y2: int
    name: str = ""
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def as_slice(self) -> Tuple[slice, slice]:
        """Return as numpy slices (y1:y2, x1:x2)."""
        return slice(self.y1, self.y2), slice(self.x1, self.x2)
    
    def extract_from(self, frame: np.ndarray) -> np.ndarray:
        """Extract ROI from frame."""
        return frame[self.y1:self.y2, self.x1:self.x2]
    
    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "name": self.name}
    
    @classmethod
    def from_dict(cls, data: dict) -> "ROI":
        return cls(**data)
    
    @classmethod
    def from_cv2_rect(cls, rect: Tuple[int, int, int, int], name: str = "") -> "ROI":
        """Create from cv2.selectROI output (x, y, w, h)."""
        x, y, w, h = rect
        return cls(x1=x, y1=y, x2=x+w, y2=y+h, name=name)


@dataclass  
class ExtractionResult:
    """Result of a metric extraction."""
    value: Any
    confidence: float = 1.0
    raw_output: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        return self.error is None and self.value is not None


class BaseExtractor(ABC):
    """Base class for all metric extractors."""
    
    name: str = "base"
    description: str = "Base extractor"
    
    def __init__(self, roi: Optional[ROI] = None):
        self.roi = roi
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize any heavy resources (e.g., OCR models). Called once."""
        self._initialized = True
    
    def cleanup(self) -> None:
        """Release resources."""
        self._initialized = False
    
    @abstractmethod
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract the metric from a frame.
        
        Args:
            frame: Full video frame (BGR format)
            
        Returns:
            ExtractionResult with the extracted value
        """
        pass
    
    def extract_from_roi(self, frame: np.ndarray) -> ExtractionResult:
        """Extract from the configured ROI."""
        if self.roi is None:
            return ExtractionResult(value=None, error="No ROI configured")
        
        roi_frame = self.roi.extract_from(frame)
        if roi_frame.size == 0:
            return ExtractionResult(value=None, error="Empty ROI")
        
        return self.extract(roi_frame)
    
    def configure(self, **kwargs) -> None:
        """Configure extractor parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_config(self) -> dict:
        """Get current configuration."""
        config = {"name": self.name}
        if self.roi:
            config["roi"] = self.roi.to_dict()
        return config
