"""Visual debugging utilities for jitter inspection."""

import cv2
import numpy as np

from video_pii.track.face_tracker import TrackState


# Color palette for track IDs
TRACK_COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Orange
    (255, 128, 0),   # Light Blue
    (128, 255, 0),   # Light Green
    (0, 128, 255),   # Salmon
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackState],
    draw_bbox: bool = True,
    draw_id: bool = True,
    draw_history: bool = False,
    bbox_thickness: int = 2,
) -> np.ndarray:
    """Draw track visualizations on a frame.
    
    Args:
        frame: Input frame (H, W, 3) in BGR format.
        tracks: List of active tracks to visualize.
        draw_bbox: Whether to draw bounding boxes.
        draw_id: Whether to draw track IDs.
        draw_history: Whether to draw bbox history trail.
        bbox_thickness: Line thickness for bounding boxes.
        
    Returns:
        Frame with overlays drawn.
    """
    if frame.size == 0 or len(tracks) == 0:
        return frame.copy()
    
    result = frame.copy()
    
    for track in tracks:
        color = get_track_color(track.track_id)
        bbox = track.bbox
        
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Draw history trail
        if draw_history and len(track.history) > 1:
            for i in range(1, len(track.history)):
                prev = track.history[i - 1]
                curr = track.history[i]
                
                # Draw line connecting bbox centers
                prev_center = (int((prev[0] + prev[2]) / 2), int((prev[1] + prev[3]) / 2))
                curr_center = (int((curr[0] + curr[2]) / 2), int((curr[1] + curr[3]) / 2))
                
                # Fade color based on history position
                alpha = (i + 1) / len(track.history)
                faded_color = tuple(int(c * alpha) for c in color)
                
                cv2.line(result, prev_center, curr_center, faded_color, 1)
        
        # Draw bounding box
        if draw_bbox:
            cv2.rectangle(result, (x1, y1), (x2, y2), color, bbox_thickness)
            
            # Draw missed frames indicator if track is being predicted
            if track.missed_frames > 0:
                cv2.putText(
                    result,
                    f"M:{track.missed_frames}",
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
        
        # Draw track ID
        if draw_id:
            label = f"ID:{track.track_id}"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            
            # Text
            cv2.putText(
                result,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    
    return result


def draw_detections(
    frame: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
    scores: list[float] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw detection bounding boxes (before tracking).
    
    Args:
        frame: Input frame.
        bboxes: List of (x1, y1, x2, y2) bounding boxes.
        scores: Optional confidence scores to display.
        color: Box color in BGR.
        thickness: Line thickness.
        
    Returns:
        Frame with detections drawn.
    """
    if frame.size == 0:
        return frame.copy()
    
    result = frame.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        if scores is not None and i < len(scores):
            label = f"{scores[i]:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
    
    return result


def create_comparison_frame(
    original: np.ndarray,
    processed: np.ndarray,
    layout: str = "horizontal",
) -> np.ndarray:
    """Create side-by-side or stacked comparison of original and processed.
    
    Args:
        original: Original frame.
        processed: Processed frame.
        layout: 'horizontal' or 'vertical'.
        
    Returns:
        Combined comparison frame.
    """
    if layout == "horizontal":
        return np.hstack([original, processed])
    else:
        return np.vstack([original, processed])

