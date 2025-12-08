"""Tests for video I/O utilities."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_pii.io.video_io import VideoMeta, VideoWriter, probe_video, read_video_frames


def create_synthetic_video(path: str, num_frames: int = 8, width: int = 64, height: int = 48, fps: float = 30.0) -> None:
    """Create a synthetic test video with colored frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 128, 128),# Gray
        (255, 255, 255),# White
    ]
    
    for i in range(num_frames):
        frame = np.full((height, width, 3), colors[i % len(colors)], dtype=np.uint8)
        writer.write(frame)
    
    writer.release()


class TestVideoMeta:
    """Tests for VideoMeta dataclass."""
    
    def test_construct_with_required_fields(self):
        meta = VideoMeta(width=1920, height=1080, fps=30.0)
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.fps == 30.0
    
    def test_optional_fields_default_to_none(self):
        meta = VideoMeta(width=640, height=480, fps=24.0)
        assert meta.frame_count is None
        assert meta.codec is None
    
    def test_construct_with_all_fields(self):
        meta = VideoMeta(width=1280, height=720, fps=60.0, frame_count=1000, codec="avc1")
        assert meta.frame_count == 1000
        assert meta.codec == "avc1"


class TestProbeVideo:
    """Tests for probe_video function."""
    
    def test_probe_returns_correct_dimensions(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        create_synthetic_video(video_path, num_frames=8, width=64, height=48)
        
        meta = probe_video(video_path)
        assert meta.width == 64
        assert meta.height == 48
    
    def test_probe_returns_positive_fps(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        create_synthetic_video(video_path, num_frames=8, fps=25.0)
        
        meta = probe_video(video_path)
        assert meta.fps > 0
    
    def test_probe_returns_frame_count(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        create_synthetic_video(video_path, num_frames=16)
        
        meta = probe_video(video_path)
        assert meta.frame_count is not None
        assert meta.frame_count >= 1
    
    def test_probe_invalid_path_raises(self):
        with pytest.raises(ValueError, match="Cannot open video"):
            probe_video("/nonexistent/path/video.mp4")


class TestReadVideoFrames:
    """Tests for read_video_frames function."""
    
    def test_yields_correct_number_of_frames(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        expected_frames = 12
        create_synthetic_video(video_path, num_frames=expected_frames)
        
        frames = list(read_video_frames(video_path))
        assert len(frames) == expected_frames
    
    def test_frames_are_numpy_arrays(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        create_synthetic_video(video_path, num_frames=4)
        
        for frame in read_video_frames(video_path):
            assert isinstance(frame, np.ndarray)
    
    def test_frames_have_correct_shape(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        create_synthetic_video(video_path, num_frames=4, width=64, height=48)
        
        for frame in read_video_frames(video_path):
            assert frame.shape == (48, 64, 3)
    
    def test_frames_are_bgr_uint8(self, tmp_path: Path):
        video_path = str(tmp_path / "test.mp4")
        create_synthetic_video(video_path, num_frames=4)
        
        for frame in read_video_frames(video_path):
            assert frame.dtype == np.uint8
    
    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="Cannot open video"):
            list(read_video_frames("/nonexistent/path/video.mp4"))


class TestVideoWriter:
    """Tests for VideoWriter class."""
    
    def test_creates_output_file(self, tmp_path: Path):
        video_path = str(tmp_path / "output.mp4")
        meta = VideoMeta(width=64, height=48, fps=30.0)
        
        with VideoWriter(video_path, meta) as writer:
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            writer.write_frame(frame)
        
        assert os.path.exists(video_path)
    
    def test_tracks_frame_count(self, tmp_path: Path):
        video_path = str(tmp_path / "output.mp4")
        meta = VideoMeta(width=64, height=48, fps=30.0)
        
        with VideoWriter(video_path, meta) as writer:
            for _ in range(5):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                writer.write_frame(frame)
            assert writer.frame_count == 5
    
    def test_wrong_frame_dimensions_raises(self, tmp_path: Path):
        video_path = str(tmp_path / "output.mp4")
        meta = VideoMeta(width=64, height=48, fps=30.0)
        
        with VideoWriter(video_path, meta) as writer:
            wrong_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with pytest.raises(ValueError, match="does not match"):
                writer.write_frame(wrong_frame)
    
    def test_write_after_close_raises(self, tmp_path: Path):
        video_path = str(tmp_path / "output.mp4")
        meta = VideoMeta(width=64, height=48, fps=30.0)
        
        writer = VideoWriter(video_path, meta)
        writer.close()
        
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="already closed"):
            writer.write_frame(frame)


class TestRoundTrip:
    """Integration test: read, write, and verify a video."""
    
    def test_round_trip_preserves_frame_count(self, tmp_path: Path):
        # Create original video
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        original_frame_count = 10
        create_synthetic_video(input_path, num_frames=original_frame_count)
        
        # Read and write
        meta = probe_video(input_path)
        with VideoWriter(output_path, meta) as writer:
            for frame in read_video_frames(input_path):
                writer.write_frame(frame)
        
        # Verify
        output_meta = probe_video(output_path)
        assert output_meta.frame_count == original_frame_count
    
    def test_round_trip_preserves_dimensions(self, tmp_path: Path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        create_synthetic_video(input_path, width=128, height=96)
        
        meta = probe_video(input_path)
        with VideoWriter(output_path, meta) as writer:
            for frame in read_video_frames(input_path):
                writer.write_frame(frame)
        
        output_meta = probe_video(output_path)
        assert output_meta.width == 128
        assert output_meta.height == 96

