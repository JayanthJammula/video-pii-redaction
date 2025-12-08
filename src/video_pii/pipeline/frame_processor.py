"""Frame processor for single-frame face anonymization.

Uses landmark-based face alignment for high-quality anonymization.
"""

import cv2
import numpy as np
from PIL import Image

from video_pii.anon.face_anon_wrapper import FaceAnonModel
from video_pii.config import PipelineConfig
from video_pii.detect.scrfd_detector import SCRFDFaceDetector
from video_pii.track.face_tracker import FaceTracker
from video_pii.utils.geometry import expand_bbox


class FrameProcessor:
    """Process individual frames for face anonymization.

    Uses face_alignment library for landmark-based extraction and
    high-quality LANCZOS4 paste-back for best results.
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
        self._face_aligner = None
        self._use_alignment = True  # Use landmark-based alignment
    
    def _get_face_aligner(self):
        """Lazy-load face_alignment module."""
        if self._face_aligner is None:
            try:
                import face_alignment
                device = 'cpu' if self.config.device == 'cpu' else self.config.device
                if device.startswith('cuda'):
                    device = 'cuda'
                self._face_aligner = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    face_detector='sfd',
                    device=device
                )
            except ImportError:
                self._use_alignment = False
                return None
        return self._face_aligner

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Process a single frame for face anonymization.

        Uses landmark-based face alignment for high-quality results.

        Args:
            frame: Input frame (H, W, 3) in BGR format.
            frame_idx: Current frame index.

        Returns:
            Processed frame with anonymized faces.
        """
        if frame.size == 0:
            return frame.copy()

        result = frame.copy()

        # Try using face_alignment for best quality
        if self._use_alignment:
            aligned_result = self._process_with_alignment(frame, frame_idx)
            if aligned_result is not None:
                return aligned_result

        # Fallback to bbox-based processing
        return self._process_with_bbox(frame, frame_idx, result)

    def _process_with_alignment(self, frame: np.ndarray, frame_idx: int) -> np.ndarray | None:
        """Process frame using landmark-based face alignment.

        Returns None if alignment fails, allowing fallback to bbox method.
        """
        import sys
        import os

        # Add face_anon_simple to path for utilities
        repo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "face_anon_simple")
        if os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from utils.extractor import extract_faces
        except ImportError:
            return None

        fa = self._get_face_aligner()
        if fa is None:
            return None

        # Convert BGR to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Extract aligned 512x512 faces with transformation matrices
        try:
            face_images, matrices = extract_faces(fa, pil_frame, 512)
        except Exception:
            return None

        if not face_images:
            return frame.copy()

        # Process each aligned face
        result_pil = pil_frame.copy()

        for i, (face_pil, matrix) in enumerate(zip(face_images, matrices)):
            # Generate anon code based on face index (for tracking consistency)
            anon_code = self.anon_model.sample_anon_code(frame_idx * 100 + i)

            # Convert PIL to BGR numpy for the model
            face_rgb = np.array(face_pil)
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

            # Generate anonymized face
            anon_bgr = self.anon_model.generate_anon_face(face_bgr, anon_code)

            # Convert back to RGB PIL
            anon_rgb = cv2.cvtColor(anon_bgr, cv2.COLOR_BGR2RGB)
            anon_pil = Image.fromarray(anon_rgb)

            # Paste back with high-quality transform
            result_pil = self._paste_aligned_face(anon_pil, result_pil, matrix)

        # Convert result back to BGR numpy
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def _paste_aligned_face(self, fg_image: Image.Image, bg_image: Image.Image,
                            rotation_matrix: np.ndarray) -> Image.Image:
        """Paste aligned face back using high-quality LANCZOS4 interpolation."""
        fg_array = np.array(fg_image)
        bg_array = np.array(bg_image)[:, :, :3].copy()

        height, width = bg_array.shape[:2]

        # Use LANCZOS4 for high-quality upscaling
        warped_fg_255 = cv2.warpAffine(
            fg_array, rotation_matrix, (width, height),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        warped_fg_0 = cv2.warpAffine(
            fg_array, rotation_matrix, (width, height),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Create mask: diff is 255 outside face, ~0 inside face
        diff = cv2.absdiff(warped_fg_255, warped_fg_0)
        mask = diff / 255.0

        # Feather edges for smooth blending
        mask_gray = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mask_blurred = cv2.GaussianBlur(mask_gray, (7, 7), 0)
        mask_feathered = np.stack([mask_blurred / 255.0] * 3, axis=-1)

        # Blend: bg where mask=1, face where mask=0
        result = mask_feathered * bg_array + warped_fg_0

        return Image.fromarray(result.astype('uint8'), 'RGB')

    def _process_with_bbox(self, frame: np.ndarray, frame_idx: int,
                           result: np.ndarray) -> np.ndarray:
        """Fallback: Process frame using simple bbox cropping."""
        height, width = frame.shape[:2]

        # Step 1: Detect faces
        detections = self.detector.detect(frame)
        detections = detections[:self.config.max_faces_per_frame]

        # Step 2: Update tracker
        tracks = self.tracker.update(detections, frame_idx)

        # Step 3: Process each tracked face
        for track in tracks:
            if track.missed_frames > 0:
                continue

            if track.anon_code is None:
                track.anon_code = self.anon_model.sample_anon_code(track.track_id)

            crop_bbox = expand_bbox(
                track.bbox,
                padding=self.config.crop_padding,
                img_width=width,
                img_height=height,
            )

            x1, y1, x2, y2 = crop_bbox
            if x2 <= x1 or y2 <= y1:
                continue

            face_patch = frame[y1:y2, x1:x2].copy()
            anon_patch = self.anon_model.generate_anon_face(face_patch, track.anon_code)

            if anon_patch.shape[:2] != (y2 - y1, x2 - x1):
                anon_patch = cv2.resize(anon_patch, (x2 - x1, y2 - y1))

            result = self._paste_with_feather(result, anon_patch, x1, y1, x2, y2)

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

