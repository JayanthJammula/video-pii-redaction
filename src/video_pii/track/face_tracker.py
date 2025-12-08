"""Face tracking across video frames with stable IDs."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from video_pii.detect.scrfd_detector import FaceDet
from video_pii.utils.geometry import iou, smooth_bbox


@dataclass
class TrackState:
    """State of a tracked face.
    
    Attributes:
        track_id: Unique identifier for this track.
        bbox: Current smoothed bounding box as numpy array [x1, y1, x2, y2].
        last_seen_frame_idx: Frame index when this track was last matched.
        missed_frames: Number of consecutive frames without a match.
        history: History of bounding boxes for this track.
        anon_code: Anonymization code for consistent identity (set by anon module).
    """
    track_id: int
    bbox: np.ndarray
    last_seen_frame_idx: int
    missed_frames: int = 0
    history: list[np.ndarray] = field(default_factory=list)
    anon_code: Any | None = None
    
    def __post_init__(self):
        if not isinstance(self.bbox, np.ndarray):
            self.bbox = np.array(self.bbox, dtype=np.float32)


class FaceTracker:
    """Track faces across frames using IoU-based matching.
    
    Assigns stable track IDs to faces and smooths bounding boxes
    to reduce jitter.
    """
    
    def __init__(
        self,
        iou_match_thresh: float = 0.4,
        max_missed_frames: int = 10,
        bbox_smoothing_alpha: float = 0.7,
        max_history: int = 30,
    ):
        """Initialize the face tracker.
        
        Args:
            iou_match_thresh: Minimum IoU to match detection to track.
            max_missed_frames: Frames before dropping unmatched track.
            bbox_smoothing_alpha: EMA smoothing factor for bboxes.
            max_history: Maximum bbox history length per track.
        """
        self.iou_match_thresh = iou_match_thresh
        self.max_missed_frames = max_missed_frames
        self.bbox_smoothing_alpha = bbox_smoothing_alpha
        self.max_history = max_history
        
        self._tracks: list[TrackState] = []
        self._next_track_id = 0
    
    def update(self, dets: list[FaceDet], frame_idx: int) -> list[TrackState]:
        """Update tracks with new detections.
        
        Args:
            dets: List of face detections for current frame.
            frame_idx: Current frame index.
            
        Returns:
            List of active TrackState objects.
        """
        # Build IoU matrix between existing tracks and new detections
        n_tracks = len(self._tracks)
        n_dets = len(dets)
        
        if n_tracks == 0 and n_dets == 0:
            return []
        
        # Match detections to tracks using greedy IoU matching
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        
        if n_tracks > 0 and n_dets > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for t_idx, track in enumerate(self._tracks):
                for d_idx, det in enumerate(dets):
                    iou_matrix[t_idx, d_idx] = iou(track.bbox, det.bbox)
            
            # Greedy matching: iteratively match highest IoU pairs
            while True:
                # Find best unmatched pair
                best_iou = 0.0
                best_t_idx = -1
                best_d_idx = -1
                
                for t_idx in range(n_tracks):
                    if t_idx in matched_tracks:
                        continue
                    for d_idx in range(n_dets):
                        if d_idx in matched_dets:
                            continue
                        if iou_matrix[t_idx, d_idx] > best_iou:
                            best_iou = iou_matrix[t_idx, d_idx]
                            best_t_idx = t_idx
                            best_d_idx = d_idx
                
                if best_iou < self.iou_match_thresh:
                    break
                
                # Match found
                matched_tracks.add(best_t_idx)
                matched_dets.add(best_d_idx)
                
                # Update track with matched detection
                track = self._tracks[best_t_idx]
                det = dets[best_d_idx]
                
                # Smooth bbox
                smoothed = smooth_bbox(track.bbox, det.bbox, self.bbox_smoothing_alpha)
                track.bbox = np.array(smoothed, dtype=np.float32)
                track.last_seen_frame_idx = frame_idx
                track.missed_frames = 0
                
                # Update history
                track.history.append(track.bbox.copy())
                if len(track.history) > self.max_history:
                    track.history.pop(0)
        
        # Handle unmatched tracks
        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                track.missed_frames += 1
        
        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(dets):
            if d_idx not in matched_dets:
                bbox_arr = np.array(det.bbox, dtype=np.float32)
                new_track = TrackState(
                    track_id=self._next_track_id,
                    bbox=bbox_arr,
                    last_seen_frame_idx=frame_idx,
                    missed_frames=0,
                    history=[bbox_arr.copy()],
                )
                self._tracks.append(new_track)
                self._next_track_id += 1
        
        # Remove tracks that have been missing too long
        self._tracks = [
            t for t in self._tracks
            if t.missed_frames <= self.max_missed_frames
        ]
        
        return self._tracks.copy()
    
    def reset(self) -> None:
        """Reset all tracks."""
        self._tracks.clear()
        self._next_track_id = 0
    
    @property
    def active_tracks(self) -> list[TrackState]:
        """Get list of currently active tracks."""
        return self._tracks.copy()

