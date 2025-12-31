"""
Image preprocessing pipeline for OCR accuracy improvement.

This module provides various preprocessing techniques optimized
for extracting text from video overlays with different font styles,
colors, and backgrounds.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import cv2
import numpy as np


class PreprocessMode(Enum):
    """Preprocessing modes for different overlay types."""
    AUTO = "auto"           # Automatically detect best settings
    LIGHT_ON_DARK = "light_on_dark"  # White/light text on dark background
    DARK_ON_LIGHT = "dark_on_light"  # Dark text on light background
    ADAPTIVE = "adaptive"    # Adaptive thresholding for mixed backgrounds


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""
    # Mode
    mode: PreprocessMode = PreprocessMode.AUTO
    
    # Resize
    scale_factor: float = 2.0  # Upscale for better OCR accuracy
    min_height: int = 32       # Minimum height after scaling
    max_height: int = 128      # Maximum height to prevent slowdown
    
    # Color processing
    convert_grayscale: bool = True
    invert_colors: bool = False  # Auto-detect if not set
    
    # Noise reduction
    denoise: bool = True
    denoise_strength: int = 10
    
    # Contrast enhancement
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    
    # Thresholding
    apply_threshold: bool = False
    threshold_value: int = 128
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    
    # Morphology (clean up text)
    apply_morphology: bool = True
    morph_kernel_size: int = 2
    
    # Sharpening
    sharpen: bool = True
    sharpen_amount: float = 1.0
    
    # Border padding (helps OCR with edge text)
    add_border: bool = True
    border_size: int = 10
    border_color: int = 255  # White for light-on-dark


class ImagePreprocessor:
    """
    Preprocessing pipeline for video overlay OCR.
    
    Applies a series of image transformations to improve OCR accuracy
    on telemetry overlays from racing videos.
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed image optimized for OCR
        """
        result = image.copy()
        
        # 1. Resize for better OCR accuracy
        result = self._resize(result)
        
        # 2. Convert to grayscale
        if self.config.convert_grayscale and len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # 3. Auto-detect if we need to invert
        if self.config.mode == PreprocessMode.AUTO:
            should_invert = self._should_invert(result)
        elif self.config.mode == PreprocessMode.DARK_ON_LIGHT:
            should_invert = True
        else:
            should_invert = self.config.invert_colors
        
        if should_invert:
            result = cv2.bitwise_not(result)
        
        # 4. Denoise
        if self.config.denoise:
            result = self._denoise(result)
        
        # 5. Enhance contrast
        if self.config.enhance_contrast:
            result = self._enhance_contrast(result)
        
        # 6. Sharpen
        if self.config.sharpen:
            result = self._sharpen(result)
        
        # 7. Thresholding (optional - can help with noisy backgrounds)
        if self.config.apply_threshold:
            result = self._threshold(result)
        
        # 8. Morphological operations
        if self.config.apply_morphology:
            result = self._apply_morphology(result)
        
        # 9. Add border padding
        if self.config.add_border:
            result = self._add_border(result)
        
        return result
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image for optimal OCR."""
        h, w = image.shape[:2]
        
        # Calculate new dimensions
        new_h = int(h * self.config.scale_factor)
        new_h = max(self.config.min_height, min(self.config.max_height, new_h))
        
        scale = new_h / h
        new_w = int(w * scale)
        
        if scale > 1:
            # Upscaling - use INTER_CUBIC for quality
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif scale < 1:
            # Downscaling - use INTER_AREA
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _should_invert(self, gray_image: np.ndarray) -> bool:
        """
        Determine if image should be inverted (dark text on light background).
        
        OCR works best with dark text on light background, so we check
        if the majority of the image is dark (meaning light text).
        """
        mean_val = np.mean(gray_image)
        # If image is mostly dark, text is likely light -> don't invert
        # If image is mostly light, text is likely dark -> invert
        return mean_val > 127
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, 
                self.config.denoise_strength, 
                self.config.denoise_strength, 7, 21
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, None, 
                self.config.denoise_strength, 7, 21
            )
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
            )
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
            )
            return clahe.apply(image)
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for sharpening."""
        amount = self.config.sharpen_amount
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened
    
    def _threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply thresholding."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.config.use_adaptive_threshold:
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                self.config.adaptive_block_size, self.config.adaptive_c
            )
        else:
            _, result = cv2.threshold(
                image, self.config.threshold_value, 255, cv2.THRESH_BINARY
            )
            return result
    
    def _apply_morphology(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up text."""
        kernel = np.ones(
            (self.config.morph_kernel_size, self.config.morph_kernel_size), 
            np.uint8
        )
        # Slight dilation to connect broken characters
        result = cv2.dilate(image, kernel, iterations=1)
        # Erosion to restore size
        result = cv2.erode(result, kernel, iterations=1)
        return result
    
    def _add_border(self, image: np.ndarray) -> np.ndarray:
        """Add padding around image."""
        return cv2.copyMakeBorder(
            image,
            self.config.border_size, self.config.border_size,
            self.config.border_size, self.config.border_size,
            cv2.BORDER_CONSTANT,
            value=self.config.border_color
        )


# Preset configurations for common scenarios
PRESETS = {
    # For typical racing game/sim telemetry overlays (white text on dark HUD)
    "racing_hud": PreprocessConfig(
        mode=PreprocessMode.LIGHT_ON_DARK,
        scale_factor=2.5,
        denoise=True,
        denoise_strength=8,
        enhance_contrast=True,
        sharpen=True,
        sharpen_amount=0.8,
        apply_threshold=False,
        apply_morphology=True,
    ),
    
    # For F1 TV style overlays
    "f1_tv": PreprocessConfig(
        mode=PreprocessMode.AUTO,
        scale_factor=3.0,
        denoise=True,
        denoise_strength=5,
        enhance_contrast=True,
        clahe_clip_limit=3.0,
        sharpen=True,
        sharpen_amount=1.2,
        apply_threshold=False,
        apply_morphology=True,
    ),
    
    # For digital readout style (like car dashboard displays)
    "digital_display": PreprocessConfig(
        mode=PreprocessMode.LIGHT_ON_DARK,
        scale_factor=2.0,
        denoise=True,
        enhance_contrast=True,
        sharpen=True,
        sharpen_amount=1.5,
        apply_threshold=True,
        use_adaptive_threshold=False,
        threshold_value=100,
        apply_morphology=True,
    ),
    
    # Conservative settings - minimal processing
    "minimal": PreprocessConfig(
        mode=PreprocessMode.AUTO,
        scale_factor=2.0,
        denoise=False,
        enhance_contrast=True,
        sharpen=False,
        apply_threshold=False,
        apply_morphology=False,
    ),
    
    # Aggressive - for difficult/noisy overlays
    "aggressive": PreprocessConfig(
        mode=PreprocessMode.AUTO,
        scale_factor=3.0,
        denoise=True,
        denoise_strength=15,
        enhance_contrast=True,
        clahe_clip_limit=4.0,
        sharpen=True,
        sharpen_amount=2.0,
        apply_threshold=True,
        use_adaptive_threshold=True,
        apply_morphology=True,
        morph_kernel_size=3,
    ),
}


def get_preprocessor(preset: str = "racing_hud") -> ImagePreprocessor:
    """Get a preprocessor with preset configuration."""
    config = PRESETS.get(preset, PRESETS["racing_hud"])
    return ImagePreprocessor(config)


def preprocess_for_ocr(image: np.ndarray, preset: str = "racing_hud") -> np.ndarray:
    """Convenience function to preprocess image for OCR."""
    preprocessor = get_preprocessor(preset)
    return preprocessor.process(image)
