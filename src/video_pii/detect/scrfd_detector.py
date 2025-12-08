"""SCRFD face detector wrapper with testable interface."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class FaceDet:
    """A single face detection.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        score: Confidence score in [0, 1].
        landmarks: Optional facial landmarks array of shape (5, 2) for 5 keypoints.
    """
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray | None = None

    def __post_init__(self):
        # Ensure bbox is a tuple of floats
        if not isinstance(self.bbox, tuple):
            self.bbox = tuple(self.bbox)


class DetectorBackend(Protocol):
    """Protocol for detector backends (real or fake)."""

    def __call__(
        self,
        frame: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], float, np.ndarray | None]]:
        """Detect faces in a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of (bbox, score, landmarks) tuples.
        """
        ...


class SCRFDFaceDetector:
    """Face detector using SCRFD model via insightface.

    Supports dependency injection of backend for testing.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
        min_face_score: float = 0.4,
        backend: DetectorBackend | None = None,
        det_size: tuple[int, int] = (640, 640),
    ):
        """Initialize the SCRFD detector.

        Args:
            model_name: InsightFace model name (e.g., 'buffalo_l', 'buffalo_sc').
            device: Compute device ('cuda', 'cpu', 'mps').
            min_face_score: Minimum score threshold for detections.
            backend: Optional backend for testing (bypasses real model).
            det_size: Detection input size (width, height).
        """
        self.device = device
        self.min_face_score = min_face_score
        self._backend = backend
        self._model = None
        self._det_size = det_size

        if backend is None:
            self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load the SCRFD model via insightface.

        Args:
            model_name: InsightFace model pack name.
        """
        try:
            from insightface.app import FaceAnalysis

            # Map device names
            providers = self._get_providers()

            # Initialize FaceAnalysis
            self._model = FaceAnalysis(
                name=model_name,
                providers=providers,
            )
            self._model.prepare(ctx_id=0, det_size=self._det_size)

        except ImportError:
            import warnings
            warnings.warn(
                "insightface not installed. Install with: pip install insightface"
            )
            self._model = None
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load SCRFD model: {e}")
            self._model = None

    def _get_providers(self) -> list[str]:
        """Get ONNX Runtime execution providers based on device."""
        if self.device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "mps":
            # MPS not directly supported by ONNX Runtime, fall back to CPU
            return ["CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    def detect(self, frame: np.ndarray) -> list[FaceDet]:
        """Detect faces in a frame.
        
        Args:
            frame: BGR image as numpy array (H, W, 3).
            
        Returns:
            List of FaceDet sorted by score (descending), filtered by min_face_score.
        """
        if frame.size == 0:
            return []
        
        height, width = frame.shape[:2]
        
        # Get raw detections from backend
        if self._backend is not None:
            raw_dets = self._backend(frame)
        elif self._model is not None:
            raw_dets = self._run_model(frame)
        else:
            # No backend or model - return empty
            return []
        
        # Process detections
        detections = []
        for bbox, score, landmarks in raw_dets:
            # Filter by score
            if score < self.min_face_score:
                continue
            
            # Clip bbox to image boundaries
            x1 = max(0.0, min(float(bbox[0]), float(width)))
            y1 = max(0.0, min(float(bbox[1]), float(height)))
            x2 = max(0.0, min(float(bbox[2]), float(width)))
            y2 = max(0.0, min(float(bbox[3]), float(height)))
            
            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            clipped_bbox = (x1, y1, x2, y2)
            detections.append(FaceDet(bbox=clipped_bbox, score=score, landmarks=landmarks))
        
        # Sort by score descending
        detections.sort(key=lambda d: d.score, reverse=True)
        
        return detections
    
    def _run_model(self, frame: np.ndarray) -> list[tuple[tuple[float, float, float, float], float, np.ndarray | None]]:
        """Run the SCRFD model via insightface.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of (bbox, score, landmarks) tuples.
        """
        if self._model is None:
            return []

        # Run face analysis
        faces = self._model.get(frame)

        results = []
        for face in faces:
            # Extract bounding box (x1, y1, x2, y2)
            bbox = tuple(float(x) for x in face.bbox)

            # Extract detection score
            score = float(face.det_score)

            # Extract landmarks if available (5 keypoints for SCRFD)
            landmarks = None
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = np.array(face.kps, dtype=np.float32)

            results.append((bbox, score, landmarks))

        return results

