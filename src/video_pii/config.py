"""Configuration models for the video PII redaction pipeline."""

from pydantic import BaseModel, Field, field_validator, model_validator


class PipelineConfig(BaseModel):
    """Configuration for the video PII redaction pipeline.

    Attributes:
        device: Compute device ('cuda', 'cpu', 'mps', 'auto').
            'auto' selects the best available device.
        min_face_score: Minimum confidence score for face detections [0, 1].
        iou_match_thresh: IoU threshold for matching detections to tracks [0, 1].
        max_missed_frames: Max frames a track can be missed before removal.
        crop_padding: Padding ratio around face bbox for cropping.
        bbox_smoothing_alpha: Smoothing factor for bbox EMA [0, 1].
            Higher = more weight to current detection.
        edge_feather_px: Pixels to feather at edges when pasting anonymized faces.
        max_faces_per_frame: Maximum number of faces to process per frame.
        debug_overlay: Whether to draw debug overlays on output.

        # Model configuration
        scrfd_model_name: InsightFace model name for SCRFD detector.
        anon_model_id: HuggingFace model ID for face anonymization (None = mock).
        anon_num_steps: Number of diffusion inference steps.
        anon_guidance_scale: Classifier-free guidance scale.
        anon_degree: Anonymization degree (higher = more different).
    """

    # General settings
    device: str = Field(default="auto", description="Compute device ('cuda', 'cpu', 'mps', 'auto')")
    min_face_score: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_match_thresh: float = Field(default=0.4, ge=0.0, le=1.0)
    max_missed_frames: int = Field(default=10, ge=0)
    crop_padding: float = Field(default=0.3, ge=0.0)
    bbox_smoothing_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    edge_feather_px: int = Field(default=8, ge=0)
    max_faces_per_frame: int = Field(default=16, ge=1)
    debug_overlay: bool = Field(default=False)

    # SCRFD detector settings
    scrfd_model_name: str = Field(
        default="buffalo_l",
        description="InsightFace model pack name (e.g., 'buffalo_l', 'buffalo_sc')"
    )

    # Face anonymization model settings
    anon_model_id: str | None = Field(
        default=None,
        description="HuggingFace model ID for face_anon_simple (None = use mock)"
    )
    anon_num_steps: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of diffusion inference steps (higher = sharper)"
    )
    anon_guidance_scale: float = Field(
        default=7.0,
        ge=0.0,
        description="Classifier-free guidance scale (higher = preserve structure better)"
    )
    anon_degree: float = Field(
        default=1.1,
        ge=0.0,
        description="Anonymization degree (1.1 = best quality, higher = more different)"
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is a recognized type."""
        valid_prefixes = ("cuda", "cpu", "mps", "auto")
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"device must start with one of {valid_prefixes}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def resolve_auto_device(self) -> "PipelineConfig":
        """Resolve 'auto' device to the best available device."""
        if self.device == "auto":
            from .utils.device import get_best_device
            object.__setattr__(self, "device", get_best_device())
        return self

    def get_resolved_device(self) -> str:
        """Get the resolved device (handles 'auto' dynamically).

        Returns:
            Actual device string ('cuda', 'mps', or 'cpu').
        """
        if self.device == "auto":
            from .utils.device import get_best_device
            return get_best_device()
        return self.device

    model_config = {"frozen": False, "extra": "forbid"}

