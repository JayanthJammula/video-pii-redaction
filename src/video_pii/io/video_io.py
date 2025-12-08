"""Video I/O utilities for reading and writing video files."""

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    """Metadata about a video file.
    
    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        frame_count: Total number of frames (may be approximate or None).
        codec: Four-character codec code (e.g., 'mp4v', 'avc1').
    """
    width: int
    height: int
    fps: float
    frame_count: int | None = None
    codec: str | None = None


def probe_video(path: str) -> VideoMeta:
    """Probe a video file and extract its metadata.
    
    Args:
        path: Path to the video file.
        
    Returns:
        VideoMeta containing width, height, fps, frame_count, and codec.
        
    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Decode fourcc int to string
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        # Clean up codec string (remove null bytes)
        codec = codec.replace("\x00", "").strip() or None
        
        return VideoMeta(
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,  # Default to 30 if fps is 0
            frame_count=frame_count if frame_count > 0 else None,
            codec=codec,
        )
    finally:
        cap.release()


def read_video_frames(path: str) -> Iterator[np.ndarray]:
    """Read frames from a video file.
    
    Yields frames as numpy arrays in BGR format (OpenCV default).
    
    Args:
        path: Path to the video file.
        
    Yields:
        np.ndarray: Frame with shape (height, width, 3) in BGR format.
        
    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


class VideoWriter:
    """Write frames to a video file.
    
    Uses OpenCV VideoWriter with mp4v codec by default.
    """
    
    def __init__(self, path: str, meta: VideoMeta, codec: str | None = None):
        """Initialize the video writer.
        
        Args:
            path: Output path for the video file.
            meta: Video metadata (width, height, fps).
            codec: Four-character codec code (default: 'mp4v').
        """
        self.path = path
        self.meta = meta
        
        # Use provided codec, fallback to meta.codec, then default to mp4v
        codec_str = codec or meta.codec or "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*codec_str[:4])
        
        self._writer = cv2.VideoWriter(
            path,
            fourcc,
            meta.fps,
            (meta.width, meta.height),
        )
        
        if not self._writer.isOpened():
            raise ValueError(f"Cannot create video writer for: {path}")
        
        self._frame_count = 0
        self._closed = False
    
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the video.
        
        Args:
            frame: Frame with shape (height, width, 3) in BGR format.
            
        Raises:
            ValueError: If writer is closed or frame dimensions don't match.
        """
        if self._closed:
            raise ValueError("VideoWriter is already closed")
        
        if frame.shape[:2] != (self.meta.height, self.meta.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match "
                f"expected ({self.meta.height}, {self.meta.width})"
            )
        
        self._writer.write(frame)
        self._frame_count += 1
    
    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count
    
    def close(self) -> None:
        """Close the video writer and finalize the file."""
        if not self._closed:
            self._writer.release()
            self._closed = True
    
    def __enter__(self) -> "VideoWriter":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

