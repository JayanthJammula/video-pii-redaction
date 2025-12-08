"""Tests for visual debug utilities."""

import numpy as np
import pytest

from video_pii.track.face_tracker import TrackState
from video_pii.utils.visual_debug import (
    create_comparison_frame,
    draw_detections,
    draw_tracks,
    get_track_color,
)


class TestGetTrackColor:
    """Tests for track color assignment."""
    
    def test_returns_tuple(self):
        color = get_track_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
    
    def test_same_id_same_color(self):
        color1 = get_track_color(5)
        color2 = get_track_color(5)
        assert color1 == color2
    
    def test_different_ids_may_differ(self):
        color1 = get_track_color(0)
        color2 = get_track_color(1)
        # Different colors expected (wraps at 10)
        assert color1 != color2
    
    def test_wraps_at_palette_size(self):
        # Colors should wrap
        color0 = get_track_color(0)
        color10 = get_track_color(10)
        assert color0 == color10


class TestDrawTracks:
    """Tests for draw_tracks function."""
    
    @pytest.fixture
    def sample_frame(self):
        return np.full((100, 100, 3), 200, dtype=np.uint8)
    
    @pytest.fixture
    def sample_track(self):
        return TrackState(
            track_id=1,
            bbox=np.array([20, 20, 60, 60], dtype=np.float32),
            last_seen_frame_idx=0,
            missed_frames=0,
            history=[np.array([20, 20, 60, 60], dtype=np.float32)],
        )
    
    def test_returns_same_shape(self, sample_frame, sample_track):
        result = draw_tracks(sample_frame, [sample_track])
        assert result.shape == sample_frame.shape
    
    def test_no_tracks_returns_copy(self, sample_frame):
        result = draw_tracks(sample_frame, [])
        np.testing.assert_array_equal(result, sample_frame)
        # Verify it's a copy
        assert result is not sample_frame
    
    def test_empty_frame_no_crash(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = draw_tracks(empty, [])
        assert result.shape == empty.shape
    
    def test_modifies_frame_when_tracks_present(self, sample_frame, sample_track):
        result = draw_tracks(sample_frame, [sample_track])
        # Frame should be modified (not equal to original)
        assert not np.array_equal(result, sample_frame)
    
    def test_does_not_modify_original(self, sample_frame, sample_track):
        original_copy = sample_frame.copy()
        draw_tracks(sample_frame, [sample_track])
        np.testing.assert_array_equal(sample_frame, original_copy)
    
    def test_multiple_tracks(self, sample_frame):
        tracks = [
            TrackState(track_id=0, bbox=np.array([10, 10, 30, 30]), last_seen_frame_idx=0),
            TrackState(track_id=1, bbox=np.array([50, 50, 80, 80]), last_seen_frame_idx=0),
        ]
        result = draw_tracks(sample_frame, tracks)
        assert result.shape == sample_frame.shape


class TestDrawDetections:
    """Tests for draw_detections function."""
    
    @pytest.fixture
    def sample_frame(self):
        return np.full((100, 100, 3), 200, dtype=np.uint8)
    
    def test_returns_same_shape(self, sample_frame):
        bboxes = [(10, 10, 50, 50)]
        result = draw_detections(sample_frame, bboxes)
        assert result.shape == sample_frame.shape
    
    def test_empty_detections_returns_copy(self, sample_frame):
        result = draw_detections(sample_frame, [])
        np.testing.assert_array_equal(result, sample_frame)
    
    def test_with_scores(self, sample_frame):
        bboxes = [(10, 10, 50, 50)]
        scores = [0.95]
        result = draw_detections(sample_frame, bboxes, scores=scores)
        assert result.shape == sample_frame.shape


class TestCreateComparisonFrame:
    """Tests for comparison frame creation."""
    
    def test_horizontal_layout(self):
        frame1 = np.zeros((50, 100, 3), dtype=np.uint8)
        frame2 = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        result = create_comparison_frame(frame1, frame2, layout="horizontal")
        
        assert result.shape == (50, 200, 3)
    
    def test_vertical_layout(self):
        frame1 = np.zeros((50, 100, 3), dtype=np.uint8)
        frame2 = np.ones((50, 100, 3), dtype=np.uint8) * 255
        
        result = create_comparison_frame(frame1, frame2, layout="vertical")
        
        assert result.shape == (100, 100, 3)

