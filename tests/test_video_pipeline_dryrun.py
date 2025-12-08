"""Dry-run integration tests for video pipeline."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_pii.anon.face_anon_wrapper import AnonCode, FaceAnonModel
from video_pii.config import PipelineConfig
from video_pii.detect.scrfd_detector import FaceDet, SCRFDFaceDetector
from video_pii.io.video_io import probe_video
from video_pii.pipeline.video_pipeline import run_video_anonymization
from video_pii.track.face_tracker import FaceTracker


def create_test_video(path: str, num_frames: int = 8, width: int = 100, height: int = 100) -> None:
    """Create a test video with a moving colored square (simulating a face)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    
    for i in range(num_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        # Draw a "face" rectangle that moves slightly each frame
        x = 30 + i * 2
        y = 30 + i
        cv2.rectangle(frame, (x, y), (x + 30, y + 30), (100, 150, 200), -1)
        writer.write(frame)
    
    writer.release()


class MockDetector(SCRFDFaceDetector):
    """Mock detector that returns faces at known positions."""
    
    def __init__(self, config: PipelineConfig):
        # Don't call super().__init__ to avoid model loading
        self.min_face_score = config.min_face_score
        self._frame_idx = 0
    
    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        # Return a detection that roughly matches where we draw the "face"
        x = 30 + self._frame_idx * 2
        y = 30 + self._frame_idx
        self._frame_idx += 1
        return [FaceDet(bbox=(x, y, x + 30, y + 30), score=0.95)]


class MockAnonModel(FaceAnonModel):
    """Mock anon model that inverts colors."""
    
    def __init__(self, config: PipelineConfig):
        # Don't call super().__init__ to avoid model loading
        self.global_seed = 42
    
    def sample_anon_code(self, track_id: int) -> AnonCode:
        return AnonCode(seed=track_id * 1000)
    
    def generate_anon_face(self, face_patch: np.ndarray, anon_code: AnonCode) -> np.ndarray:
        # Simple inversion for testing
        return 255 - face_patch


def mock_detector_factory(config: PipelineConfig) -> SCRFDFaceDetector:
    return MockDetector(config)


def mock_tracker_factory(config: PipelineConfig) -> FaceTracker:
    return FaceTracker(
        iou_match_thresh=config.iou_match_thresh,
        max_missed_frames=config.max_missed_frames,
        bbox_smoothing_alpha=config.bbox_smoothing_alpha,
    )


def mock_anon_model_factory(config: PipelineConfig) -> FaceAnonModel:
    return MockAnonModel(config)


class TestVideoPipelineDryRun:
    """Dry-run integration tests with mocked ML models."""
    
    def test_output_file_created(self, tmp_path: Path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        
        create_test_video(input_path, num_frames=4)
        
        config = PipelineConfig(edge_feather_px=0)
        run_video_anonymization(
            input_path=input_path,
            output_path=output_path,
            config=config,
            detector_factory=mock_detector_factory,
            tracker_factory=mock_tracker_factory,
            anon_model_factory=mock_anon_model_factory,
        )
        
        assert os.path.exists(output_path)
    
    def test_output_has_same_frame_count(self, tmp_path: Path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        num_frames = 8
        
        create_test_video(input_path, num_frames=num_frames)
        
        config = PipelineConfig(edge_feather_px=0)
        result = run_video_anonymization(
            input_path=input_path,
            output_path=output_path,
            config=config,
            detector_factory=mock_detector_factory,
            tracker_factory=mock_tracker_factory,
            anon_model_factory=mock_anon_model_factory,
        )
        
        assert result["frames_processed"] == num_frames
        
        # Verify output is readable
        output_meta = probe_video(output_path)
        assert output_meta.frame_count == num_frames
    
    def test_output_has_same_dimensions(self, tmp_path: Path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        
        create_test_video(input_path, width=120, height=80)
        
        config = PipelineConfig(edge_feather_px=0)
        run_video_anonymization(
            input_path=input_path,
            output_path=output_path,
            config=config,
            detector_factory=mock_detector_factory,
            tracker_factory=mock_tracker_factory,
            anon_model_factory=mock_anon_model_factory,
        )
        
        input_meta = probe_video(input_path)
        output_meta = probe_video(output_path)
        
        assert output_meta.width == input_meta.width
        assert output_meta.height == input_meta.height
    
    def test_progress_callback_called(self, tmp_path: Path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        
        create_test_video(input_path, num_frames=4)
        
        callback_calls = []
        def progress(frame_idx, total):
            callback_calls.append((frame_idx, total))
        
        config = PipelineConfig(edge_feather_px=0)
        run_video_anonymization(
            input_path=input_path,
            output_path=output_path,
            config=config,
            detector_factory=mock_detector_factory,
            tracker_factory=mock_tracker_factory,
            anon_model_factory=mock_anon_model_factory,
            progress_callback=progress,
        )
        
        assert len(callback_calls) == 4
        assert callback_calls[0][0] == 0
        assert callback_calls[3][0] == 3
    
    def test_no_exception_with_default_config(self, tmp_path: Path):
        input_path = str(tmp_path / "input.mp4")
        output_path = str(tmp_path / "output.mp4")
        
        create_test_video(input_path, num_frames=2)
        
        # Should not raise any exception
        run_video_anonymization(
            input_path=input_path,
            output_path=output_path,
            detector_factory=mock_detector_factory,
            tracker_factory=mock_tracker_factory,
            anon_model_factory=mock_anon_model_factory,
        )

