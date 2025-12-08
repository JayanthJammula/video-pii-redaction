"""Tests for face tracker."""

import numpy as np
import pytest

from video_pii.detect.scrfd_detector import FaceDet
from video_pii.track.face_tracker import FaceTracker, TrackState


class TestTrackState:
    """Tests for TrackState dataclass."""
    
    def test_construct_with_tuple_bbox(self):
        track = TrackState(
            track_id=0,
            bbox=(10, 10, 50, 50),
            last_seen_frame_idx=0,
        )
        assert isinstance(track.bbox, np.ndarray)
        assert track.bbox.tolist() == [10, 10, 50, 50]
    
    def test_anon_code_defaults_to_none(self):
        track = TrackState(track_id=0, bbox=(10, 10, 50, 50), last_seen_frame_idx=0)
        assert track.anon_code is None


class TestFaceTrackerSingleFace:
    """Tests for tracking a single face."""
    
    def test_new_detection_creates_track(self):
        tracker = FaceTracker()
        det = FaceDet(bbox=(10, 10, 50, 50), score=0.9)
        
        tracks = tracker.update([det], frame_idx=0)
        
        assert len(tracks) == 1
        assert tracks[0].track_id == 0
    
    def test_same_face_keeps_same_id(self):
        tracker = FaceTracker(iou_match_thresh=0.3, bbox_smoothing_alpha=0.7)
        
        # Frame 0
        tracks = tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        first_id = tracks[0].track_id
        
        # Frame 1 - slightly moved
        tracks = tracker.update([FaceDet(bbox=(12, 12, 52, 52), score=0.9)], frame_idx=1)
        
        assert len(tracks) == 1
        assert tracks[0].track_id == first_id
    
    def test_bbox_is_smoothed(self):
        # Use low iou_match_thresh so boxes can match even with movement
        tracker = FaceTracker(bbox_smoothing_alpha=0.5, iou_match_thresh=0.1)

        # Frame 0: 40x40 box at origin
        tracker.update([FaceDet(bbox=(0, 0, 40, 40), score=0.9)], frame_idx=0)

        # Frame 1: box shifted but still overlapping
        # (10, 10, 50, 50) overlaps with (0, 0, 40, 40)
        tracks = tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=1)

        # Smoothed bbox should be between old and new
        # With alpha=0.5: (0+10)/2=5, (40+50)/2=45
        assert tracks[0].bbox[0] == pytest.approx(5.0)
        assert tracks[0].bbox[2] == pytest.approx(45.0)
    
    def test_history_updated(self):
        tracker = FaceTracker()
        
        for i in range(5):
            tracks = tracker.update([FaceDet(bbox=(10+i, 10+i, 50+i, 50+i), score=0.9)], frame_idx=i)
        
        assert len(tracks[0].history) == 5


class TestFaceTrackerTrackLifecycle:
    """Tests for track lifecycle (creation, persistence, removal)."""
    
    def test_track_missed_frames_increments(self):
        tracker = FaceTracker(max_missed_frames=5)
        
        # Create track
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        
        # No detection in next frame
        tracks = tracker.update([], frame_idx=1)
        
        assert len(tracks) == 1
        assert tracks[0].missed_frames == 1
    
    def test_track_removed_after_max_missed(self):
        tracker = FaceTracker(max_missed_frames=3)
        
        # Create track
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        
        # Miss 4 frames (> max_missed_frames)
        for i in range(1, 5):
            tracks = tracker.update([], frame_idx=i)
        
        assert len(tracks) == 0
    
    def test_track_persists_until_max_missed(self):
        tracker = FaceTracker(max_missed_frames=3)
        
        # Create track
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        
        # Miss exactly max_missed_frames (should still be present)
        for i in range(1, 4):
            tracks = tracker.update([], frame_idx=i)
        
        assert len(tracks) == 1
        assert tracks[0].missed_frames == 3
    
    def test_missed_frames_reset_on_match(self):
        tracker = FaceTracker(max_missed_frames=5)
        
        # Create and then miss
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        tracker.update([], frame_idx=1)
        tracker.update([], frame_idx=2)
        
        # Re-detect
        tracks = tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=3)
        
        assert tracks[0].missed_frames == 0


class TestFaceTrackerMultipleFaces:
    """Tests for tracking multiple faces."""
    
    def test_multiple_faces_get_unique_ids(self):
        tracker = FaceTracker()
        
        dets = [
            FaceDet(bbox=(10, 10, 50, 50), score=0.9),
            FaceDet(bbox=(100, 100, 140, 140), score=0.8),
        ]
        tracks = tracker.update(dets, frame_idx=0)
        
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id
    
    def test_ids_persist_across_frames(self):
        tracker = FaceTracker(iou_match_thresh=0.3)
        
        # Frame 0
        dets0 = [
            FaceDet(bbox=(10, 10, 50, 50), score=0.9),
            FaceDet(bbox=(100, 100, 140, 140), score=0.8),
        ]
        tracks0 = tracker.update(dets0, frame_idx=0)
        ids0 = {t.track_id for t in tracks0}
        
        # Frame 1 - slightly moved
        dets1 = [
            FaceDet(bbox=(12, 12, 52, 52), score=0.9),
            FaceDet(bbox=(102, 102, 142, 142), score=0.8),
        ]
        tracks1 = tracker.update(dets1, frame_idx=1)
        ids1 = {t.track_id for t in tracks1}
        
        assert ids0 == ids1
    
    def test_new_face_gets_new_id(self):
        tracker = FaceTracker()
        
        # Frame 0 - one face
        tracks0 = tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        id0 = tracks0[0].track_id
        
        # Frame 1 - new face appears
        dets1 = [
            FaceDet(bbox=(10, 10, 50, 50), score=0.9),  # Existing
            FaceDet(bbox=(200, 200, 240, 240), score=0.8),  # New
        ]
        tracks1 = tracker.update(dets1, frame_idx=1)
        
        ids = [t.track_id for t in tracks1]
        assert id0 in ids
        assert len(set(ids)) == 2  # Two unique IDs


class TestFaceTrackerReset:
    """Tests for tracker reset."""
    
    def test_reset_clears_tracks(self):
        tracker = FaceTracker()
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        
        tracker.reset()
        
        assert len(tracker.active_tracks) == 0
    
    def test_reset_resets_id_counter(self):
        tracker = FaceTracker()
        tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        
        tracker.reset()
        tracks = tracker.update([FaceDet(bbox=(10, 10, 50, 50), score=0.9)], frame_idx=0)
        
        assert tracks[0].track_id == 0

