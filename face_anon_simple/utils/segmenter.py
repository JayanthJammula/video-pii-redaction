import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for head segmentation. Install it via pip install mediapipe."
    ) from exc


class HeadSegmenter:
    """Wrapper around MediaPipe Selfie Segmentation to get person masks."""

    def __init__(self, model_selection: int = 1, threshold: float = 0.3, dilate_kernel: int = 15):
        self.threshold = threshold
        self.dilate_kernel = dilate_kernel
        self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

    def close(self):
        self._seg.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a uint8 mask (H x W) highlighting the foreground person."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._seg.process(frame_rgb)
        if result.segmentation_mask is None:
            return np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        mask = (result.segmentation_mask > self.threshold).astype(np.uint8) * 255
        if self.dilate_kernel and self.dilate_kernel > 0:
            k = self.dilate_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
