"""Frame processor for single-frame face anonymization."""

import cv2
import numpy as np

from video_pii.anon.face_anon_wrapper import FaceAnonModel
from video_pii.config import PipelineConfig
from video_pii.detect.scrfd_detector import SCRFDFaceDetector
from video_pii.track.face_tracker import FaceTracker
from video_pii.utils.geometry import expand_bbox


class FrameProcessor:
    """Process individual frames for face anonymization.
    
    Orchestrates detection, tracking, and anonymization with
    jitter reduction techniques.
    """
    
    def __init__(
        self,
        detector: SCRFDFaceDetector,
        tracker: FaceTracker,
        anon_model: FaceAnonModel,
        config: PipelineConfig,
    ):
        """Initialize the frame processor.
        
        Args:
            detector: Face detector instance.
            tracker: Face tracker instance.
            anon_model: Face anonymization model.
            config: Pipeline configuration.
        """
        self.detector = detector
        self.tracker = tracker
        self.anon_model = anon_model
        self.config = config
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a single frame for face anonymization.
        
        Args:
            frame: Input frame (H, W, 3) in BGR format.
            frame_idx: Current frame index.
            
        Returns:
            Processed frame with anonymized faces.
        """
        if frame.size == 0:
            return frame.copy()
        
        height, width = frame.shape[:2]
        result = frame.copy()
        
        # Step 1: Detect faces
        detections = self.detector.detect(frame)
        
        # Limit number of faces
        detections = detections[:self.config.max_faces_per_frame]
        
        # Step 2: Update tracker
        tracks = self.tracker.update(detections, frame_idx)
        
        # Step 3: Process each tracked face
        for track in tracks:
            # Skip if track wasn't matched this frame (bbox might be stale)
            if track.missed_frames > 0:
                continue
            
            # Assign anon code if not already set
            if track.anon_code is None:
                track.anon_code = self.anon_model.sample_anon_code(track.track_id)
            
            # Compute expanded crop region
            crop_bbox = expand_bbox(
                track.bbox,
                padding=self.config.crop_padding,
                img_width=width,
                img_height=height,
            )
            
            x1, y1, x2, y2 = crop_bbox
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract face patch
            face_patch = frame[y1:y2, x1:x2].copy()
            
            # Generate anonymized face
            anon_patch = self.anon_model.generate_anon_face(face_patch, track.anon_code)
            
            # Resize anon_patch to match crop region if needed
            if anon_patch.shape[:2] != (y2 - y1, x2 - x1):
                anon_patch = cv2.resize(anon_patch, (x2 - x1, y2 - y1))
            
            # Paste with edge feathering
            result = self._paste_with_feather(
                result, anon_patch, x1, y1, x2, y2
            )
        
        return result
    
    def _paste_with_feather(
        self,
        frame: np.ndarray,
        patch: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """Paste a patch onto frame with edge feathering.
        
        Args:
            frame: Target frame.
            patch: Patch to paste.
            x1, y1, x2, y2: Coordinates where to paste.
            
        Returns:
            Frame with patch pasted.
        """
        feather_px = self.config.edge_feather_px
        patch_h, patch_w = patch.shape[:2]
        
        if feather_px <= 0 or patch_h < 2 * feather_px or patch_w < 2 * feather_px:
            # No feathering - direct paste
            frame[y1:y2, x1:x2] = patch
            return frame
        
        # Create feather mask
        mask = np.ones((patch_h, patch_w), dtype=np.float32)
        
        # Feather edges
        for i in range(feather_px):
            alpha = (i + 1) / (feather_px + 1)
            # Top edge
            mask[i, :] = min(mask[i, 0], alpha)
            # Bottom edge
            mask[patch_h - 1 - i, :] = min(mask[patch_h - 1 - i, 0], alpha)
            # Left edge
            mask[:, i] = np.minimum(mask[:, i], alpha)
            # Right edge
            mask[:, patch_w - 1 - i] = np.minimum(mask[:, patch_w - 1 - i], alpha)
        
        # Apply blending
        mask_3d = mask[:, :, np.newaxis]
        original = frame[y1:y2, x1:x2].astype(np.float32)
        new_patch = patch.astype(np.float32)
        
        blended = mask_3d * new_patch + (1 - mask_3d) * original
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return frame
    
    def reset(self) -> None:
        """Reset processor state (tracker)."""
        self.tracker.reset()

