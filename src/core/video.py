"""Video handling utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Video metadata."""
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    
    @classmethod
    def from_capture(cls, cap: cv2.VideoCapture, path: str = "") -> "VideoInfo":
        return cls(
            path=path,
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        )


class VideoReader:
    """Video file reader with frame access utilities."""
    
    def __init__(self, path: str):
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
    
    def open(self) -> bool:
        """Open the video file."""
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._info = VideoInfo.from_capture(self._cap, self.path)
        return True
    
    def close(self) -> None:
        """Release video resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __enter__(self) -> "VideoReader":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def info(self) -> Optional[VideoInfo]:
        return self._info
    
    @property
    def fps(self) -> float:
        return self._info.fps if self._info else 30.0
    
    @property
    def frame_count(self) -> int:
        return self._info.frame_count if self._info else 0
    
    def read_frame(self, index: Optional[int] = None) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a specific frame or the next frame.
        
        Args:
            index: Frame index to read. If None, reads next frame.
            
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_open:
            return False, None
        
        if index is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        
        ret, frame = self._cap.read()
        return ret, frame if ret else None
    
    def get_frame_at_time(self, time_sec: float) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame at specific time in seconds."""
        frame_index = int(time_sec * self.fps)
        return self.read_frame(frame_index)
    
    def frame_to_time(self, frame_index: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_index / self.fps
    
    def time_to_frame(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.fps)
    
    def iter_frames(self, start: int = 0, end: Optional[int] = None, 
                    step: int = 1) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Iterate over frames.
        
        Args:
            start: Starting frame index
            end: Ending frame index (exclusive). None = all frames
            step: Frame step (1 = every frame, 2 = every other, etc.)
            
        Yields:
            Tuple of (frame_index, time_seconds, frame)
        """
        if not self.is_open:
            return
        
        end = end if end is not None else self.frame_count
        
        for idx in range(start, end, step):
            ret, frame = self.read_frame(idx)
            if not ret:
                break
            yield idx, self.frame_to_time(idx), frame
    
    def get_thumbnail(self, time_sec: float = 0, max_size: int = 640) -> Optional[np.ndarray]:
        """Get a thumbnail image from the video.
        
        Args:
            time_sec: Time position for thumbnail
            max_size: Maximum dimension (width or height)
            
        Returns:
            Resized frame or None
        """
        ret, frame = self.get_frame_at_time(time_sec)
        if not ret or frame is None:
            return None
        
        h, w = frame.shape[:2]
        scale = max_size / max(w, h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return frame
