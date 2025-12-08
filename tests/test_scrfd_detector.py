"""Tests for SCRFD face detector wrapper."""

import numpy as np
import pytest

from video_pii.detect.scrfd_detector import FaceDet, SCRFDFaceDetector


class FakeBackend:
    """Fake backend for testing that returns predetermined detections."""
    
    def __init__(self, detections: list[tuple[tuple[float, float, float, float], float, np.ndarray | None]]):
        """Initialize with predetermined detections.
        
        Args:
            detections: List of (bbox, score, landmarks) tuples to return.
        """
        self.detections = detections
        self.call_count = 0
    
    def __call__(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float, np.ndarray | None]]:
        self.call_count += 1
        return self.detections


class TestFaceDet:
    """Tests for FaceDet dataclass."""
    
    def test_construct_with_required_fields(self):
        det = FaceDet(bbox=(10.0, 20.0, 100.0, 150.0), score=0.95)
        assert det.bbox == (10.0, 20.0, 100.0, 150.0)
        assert det.score == 0.95
    
    def test_landmarks_default_to_none(self):
        det = FaceDet(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9)
        assert det.landmarks is None
    
    def test_construct_with_landmarks(self):
        landmarks = np.array([[30, 40], [50, 40], [40, 60], [35, 80], [45, 80]])
        det = FaceDet(bbox=(10.0, 20.0, 100.0, 150.0), score=0.9, landmarks=landmarks)
        assert det.landmarks is not None
        assert det.landmarks.shape == (5, 2)
    
    def test_bbox_converted_to_tuple(self):
        det = FaceDet(bbox=[10.0, 20.0, 100.0, 150.0], score=0.9)
        assert isinstance(det.bbox, tuple)


class TestSCRFDFaceDetectorWithFakeBackend:
    """Tests for SCRFDFaceDetector using fake backend."""
    
    def test_detect_returns_correct_count(self):
        fake_dets = [
            ((10, 10, 50, 50), 0.9, None),
            ((100, 100, 150, 150), 0.8, None),
        ]
        backend = FakeBackend(fake_dets)
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert len(detections) == 2
    
    def test_detect_filters_by_score(self):
        fake_dets = [
            ((10, 10, 50, 50), 0.9, None),
            ((100, 100, 150, 150), 0.3, None),  # Below threshold
        ]
        backend = FakeBackend(fake_dets)
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert len(detections) == 1
        assert detections[0].score == 0.9
    
    def test_detect_sorts_by_score_descending(self):
        fake_dets = [
            ((10, 10, 50, 50), 0.7, None),
            ((100, 100, 150, 150), 0.95, None),
            ((60, 60, 90, 90), 0.85, None),
        ]
        backend = FakeBackend(fake_dets)
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        scores = [d.score for d in detections]
        assert scores == sorted(scores, reverse=True)
        assert scores == [0.95, 0.85, 0.7]
    
    def test_detect_clips_bbox_to_image_bounds(self):
        # Bbox extends outside 100x100 image
        fake_dets = [
            ((-10, -5, 50, 50), 0.9, None),   # Negative coords
            ((80, 80, 150, 200), 0.8, None),  # Extends beyond image
        ]
        backend = FakeBackend(fake_dets)
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        # First detection clipped
        assert detections[0].bbox[0] >= 0  # x1
        assert detections[0].bbox[1] >= 0  # y1
        
        # Second detection clipped
        assert detections[1].bbox[2] <= 100  # x2
        assert detections[1].bbox[3] <= 100  # y2
    
    def test_detect_skips_degenerate_boxes(self):
        fake_dets = [
            ((50, 50, 50, 100), 0.9, None),  # Zero width
            ((50, 50, 100, 50), 0.9, None),  # Zero height
            ((60, 60, 50, 100), 0.9, None),  # Negative width (x2 < x1)
            ((10, 10, 50, 50), 0.8, None),   # Valid
        ]
        backend = FakeBackend(fake_dets)
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert len(detections) == 1
        assert detections[0].bbox == (10.0, 10.0, 50.0, 50.0)
    
    def test_detect_empty_frame(self):
        backend = FakeBackend([])
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert detections == []
    
    def test_detect_no_faces(self):
        backend = FakeBackend([])
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert detections == []
    
    def test_backend_called_once(self):
        fake_dets = [((10, 10, 50, 50), 0.9, None)]
        backend = FakeBackend(fake_dets)
        detector = SCRFDFaceDetector(backend=backend, min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detector.detect(frame)
        
        assert backend.call_count == 1


class TestSCRFDFaceDetectorNoBackend:
    """Tests for detector without backend (no model loaded)."""
    
    def test_detect_returns_empty_without_backend_or_model(self):
        detector = SCRFDFaceDetector(min_face_score=0.5)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert detections == []

