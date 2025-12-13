"""
Simple IoU-based Face Tracker

Tracks faces across frames using bounding box IoU (Intersection over Union).
Assigns persistent IDs to each tracked face for consistent anonymization.
"""

import numpy as np
from typing import List, Tuple, Dict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def landmarks_to_bbox(landmarks: np.ndarray, padding_ratio: float = 0.3) -> np.ndarray:
    """Convert 68-point landmarks to bounding box.
    
    Args:
        landmarks: (68, 2) array of facial landmarks
        padding_ratio: Extra padding (as a fraction of width/height) to apply

    Returns:
        [x1, y1, x2, y2] bounding box
    """
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    x1, y1, x2, y2 = x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()
    width = x2 - x1
    height = y2 - y1

    # Expand bbox on all sides by the requested padding percentage.
    x_pad = width * padding_ratio
    y_pad = height * padding_ratio

    return np.array([x1 - x_pad, y1 - y_pad, x2 + x_pad, y2 + y_pad])


class FaceTracker:
    """Simple IoU-based face tracker.
    
    Maintains persistent face IDs across frames by matching faces
    based on bounding box overlap (IoU).
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        """Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU to consider a match
            max_lost_frames: Remove track after this many frames without detection
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 0
        self.tracks: Dict[int, dict] = {}  # track_id -> {bbox, lost_frames}
    
    def update(self, landmarks_list: List[np.ndarray]) -> List[int]:
        """Update tracker with new detections.
        
        Args:
            landmarks_list: List of (68, 2) landmark arrays for each detected face
        
        Returns:
            List of track IDs corresponding to each detection (same order)
        """
        if landmarks_list is None or len(landmarks_list) == 0:
            # No detections - increment lost frames for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]
            return []
        
        # Convert landmarks to bboxes
        current_bboxes = [landmarks_to_bbox(lm) for lm in landmarks_list]
        
        # Match detections to existing tracks using IoU
        assigned_ids = [None] * len(current_bboxes)
        used_tracks = set()
        
        # Compute IoU matrix
        if self.tracks:
            track_ids = list(self.tracks.keys())
            track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
            
            # Greedy matching: assign highest IoU first
            iou_pairs = []
            for det_idx, det_bbox in enumerate(current_bboxes):
                for track_idx, track_bbox in enumerate(track_bboxes):
                    iou = compute_iou(det_bbox, track_bbox)
                    if iou >= self.iou_threshold:
                        iou_pairs.append((iou, det_idx, track_idx))
            
            # Sort by IoU descending
            iou_pairs.sort(reverse=True, key=lambda x: x[0])
            
            for iou, det_idx, track_idx in iou_pairs:
                track_id = track_ids[track_idx]
                if assigned_ids[det_idx] is None and track_id not in used_tracks:
                    assigned_ids[det_idx] = track_id
                    used_tracks.add(track_id)
                    # Update track
                    self.tracks[track_id]['bbox'] = current_bboxes[det_idx]
                    self.tracks[track_id]['lost_frames'] = 0
        
        # Create new tracks for unmatched detections
        for det_idx, track_id in enumerate(assigned_ids):
            if track_id is None:
                new_id = self.next_id
                self.next_id += 1
                assigned_ids[det_idx] = new_id
                self.tracks[new_id] = {
                    'bbox': current_bboxes[det_idx],
                    'lost_frames': 0
                }
        
        # Increment lost frames for unmatched tracks
        for track_id in self.tracks:
            if track_id not in used_tracks:
                self.tracks[track_id]['lost_frames'] += 1
        
        # Remove tracks that have been lost too long
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                del self.tracks[track_id]
        
        return assigned_ids
    
    def get_seed_for_track(self, track_id: int, base_seed: int = 42) -> int:
        """Get a unique seed for a track ID.
        
        Args:
            track_id: The track ID
            base_seed: Base seed value
        
        Returns:
            Unique seed for this track (base_seed + track_id * 1000)
        """
        return base_seed + track_id * 1000
