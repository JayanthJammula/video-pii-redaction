"""Tests for frame processor."""

import numpy as np
import pytest

from video_pii.anon.face_anon_wrapper import AnonCode, FaceAnonModel
from video_pii.config import PipelineConfig
from video_pii.detect.scrfd_detector import FaceDet, SCRFDFaceDetector
from video_pii.pipeline.frame_processor import FrameProcessor
from video_pii.track.face_tracker import FaceTracker


class MockDetector:
    """Mock detector that returns predetermined detections."""
    
    def __init__(self, detections_per_frame: dict[int, list[FaceDet]] | None = None):
        self.detections_per_frame = detections_per_frame or {}
        self.call_count = 0
        self._frame_idx = 0
    
    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        self.call_count += 1
        dets = self.detections_per_frame.get(self._frame_idx, [])
        self._frame_idx += 1
        return dets


class MockAnonModel:
    """Mock anon model that tracks calls and applies simple transform."""
    
    def __init__(self):
        self.sample_calls = []
        self.generate_calls = []
    
    def sample_anon_code(self, track_id: int) -> AnonCode:
        self.sample_calls.append(track_id)
        return AnonCode(seed=track_id * 1000)
    
    def generate_anon_face(self, face_patch: np.ndarray, anon_code: AnonCode) -> np.ndarray:
        self.generate_calls.append((face_patch.shape, anon_code.seed))
        # Return inverted colors for easy verification
        return 255 - face_patch


class TestFrameProcessorBasic:
    """Basic tests for FrameProcessor."""
    
    @pytest.fixture
    def config(self):
        return PipelineConfig(
            min_face_score=0.4,
            crop_padding=0.1,
            edge_feather_px=0,  # Disable feathering for simpler tests
        )
    
    def test_no_faces_returns_unchanged(self, config):
        detector = MockDetector({})
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = processor.process_frame(frame, frame_idx=0)
        
        np.testing.assert_array_equal(result, frame)
    
    def test_detector_called(self, config):
        detector = MockDetector({})
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor.process_frame(frame, frame_idx=0)
        
        assert detector.call_count == 1
    
    def test_face_region_modified(self, config):
        # Detection in center of 100x100 image
        detections = {0: [FaceDet(bbox=(30, 30, 70, 70), score=0.9)]}
        detector = MockDetector(detections)
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = processor.process_frame(frame, frame_idx=0)
        
        # Face region should be modified (inverted by mock)
        face_region = result[30:70, 30:70]
        # After inversion of 100, expect ~155
        assert not np.all(face_region == 100)
    
    def test_region_outside_face_unchanged(self, config):
        # Small detection in corner
        detections = {0: [FaceDet(bbox=(80, 80, 95, 95), score=0.9)]}
        detector = MockDetector(detections)
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = processor.process_frame(frame, frame_idx=0)
        
        # Far corner should be unchanged
        corner = result[0:10, 0:10]
        np.testing.assert_array_equal(corner, 100)


class TestFrameProcessorTracking:
    """Tests for tracking integration."""
    
    @pytest.fixture
    def config(self):
        return PipelineConfig(
            min_face_score=0.4,
            crop_padding=0.0,
            edge_feather_px=0,
        )
    
    def test_same_track_same_anon_code(self, config):
        # Same face position in both frames
        detections = {
            0: [FaceDet(bbox=(40, 40, 60, 60), score=0.9)],
            1: [FaceDet(bbox=(41, 41, 61, 61), score=0.9)],  # Slight movement
        }
        detector = MockDetector(detections)
        tracker = FaceTracker(iou_match_thresh=0.3)
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        
        processor.process_frame(frame, frame_idx=0)
        processor.process_frame(frame, frame_idx=1)
        
        # Should only sample anon code once for same track
        assert len(anon_model.sample_calls) == 1
    
    def test_different_tracks_different_anon_codes(self, config):
        # Two different faces
        detections = {
            0: [
                FaceDet(bbox=(10, 10, 30, 30), score=0.9),
                FaceDet(bbox=(70, 70, 90, 90), score=0.8),
            ]
        }
        detector = MockDetector(detections)
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        processor.process_frame(frame, frame_idx=0)
        
        # Should sample two different anon codes
        assert len(anon_model.sample_calls) == 2
        assert anon_model.sample_calls[0] != anon_model.sample_calls[1]


class TestFrameProcessorEdgeCases:
    """Edge case tests."""
    
    def test_empty_frame(self):
        config = PipelineConfig()
        detector = MockDetector({})
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        result = processor.process_frame(empty_frame, frame_idx=0)
        
        assert result.shape == empty_frame.shape
    
    def test_max_faces_per_frame_limit(self):
        config = PipelineConfig(max_faces_per_frame=2, edge_feather_px=0, crop_padding=0.0)
        
        # 5 detections but max is 2
        detections = {0: [
            FaceDet(bbox=(10, 10, 20, 20), score=0.9),
            FaceDet(bbox=(30, 30, 40, 40), score=0.8),
            FaceDet(bbox=(50, 50, 60, 60), score=0.7),
            FaceDet(bbox=(70, 70, 80, 80), score=0.6),
            FaceDet(bbox=(85, 85, 95, 95), score=0.5),
        ]}
        detector = MockDetector(detections)
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        processor.process_frame(frame, frame_idx=0)
        
        # Only 2 faces should be processed
        assert len(anon_model.generate_calls) == 2
    
    def test_reset_clears_tracker(self):
        config = PipelineConfig()
        detector = MockDetector({})
        tracker = FaceTracker()
        anon_model = MockAnonModel()
        processor = FrameProcessor(detector, tracker, anon_model, config)
        
        # Add some state to tracker
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        assert len(tracker.active_tracks) == 1
        
        processor.reset()
        
        assert len(tracker.active_tracks) == 0

