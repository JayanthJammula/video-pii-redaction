"""Tests for PipelineConfig."""

import pytest
from pydantic import ValidationError

from video_pii.config import PipelineConfig


class TestPipelineConfigDefaults:
    """Test that PipelineConfig constructs with expected defaults."""

    def test_default_construction(self):
        """Config should construct with no arguments."""
        config = PipelineConfig()
        assert config is not None

    def test_default_device_is_auto_resolved(self):
        """Default device is 'auto' which resolves to best available."""
        config = PipelineConfig()
        # 'auto' gets resolved to best available device (cuda, mps, or cpu)
        assert config.device in ("cuda", "mps", "cpu")

    def test_default_min_face_score(self):
        config = PipelineConfig()
        assert config.min_face_score == 0.4

    def test_default_iou_match_thresh(self):
        config = PipelineConfig()
        assert config.iou_match_thresh == 0.4

    def test_default_max_missed_frames(self):
        config = PipelineConfig()
        assert config.max_missed_frames == 10

    def test_default_crop_padding(self):
        config = PipelineConfig()
        assert config.crop_padding == 0.3

    def test_default_bbox_smoothing_alpha(self):
        config = PipelineConfig()
        assert config.bbox_smoothing_alpha == 0.7

    def test_default_edge_feather_px(self):
        config = PipelineConfig()
        assert config.edge_feather_px == 8

    def test_default_max_faces_per_frame(self):
        config = PipelineConfig()
        assert config.max_faces_per_frame == 16

    def test_default_debug_overlay(self):
        config = PipelineConfig()
        assert config.debug_overlay is False


class TestPipelineConfigOverrides:
    """Test that PipelineConfig fields can be overridden."""

    def test_override_device(self):
        config = PipelineConfig(device="cpu")
        assert config.device == "cpu"

    def test_override_device_cuda_index(self):
        config = PipelineConfig(device="cuda:1")
        assert config.device == "cuda:1"

    def test_override_device_mps(self):
        config = PipelineConfig(device="mps")
        assert config.device == "mps"

    def test_override_min_face_score(self):
        config = PipelineConfig(min_face_score=0.8)
        assert config.min_face_score == 0.8

    def test_override_iou_match_thresh(self):
        config = PipelineConfig(iou_match_thresh=0.5)
        assert config.iou_match_thresh == 0.5

    def test_override_max_missed_frames(self):
        config = PipelineConfig(max_missed_frames=5)
        assert config.max_missed_frames == 5

    def test_override_multiple_fields(self):
        config = PipelineConfig(
            device="cpu",
            min_face_score=0.6,
            max_faces_per_frame=8,
            debug_overlay=True,
        )
        assert config.device == "cpu"
        assert config.min_face_score == 0.6
        assert config.max_faces_per_frame == 8
        assert config.debug_overlay is True


class TestPipelineConfigValidation:
    """Test validation of PipelineConfig fields."""

    def test_min_face_score_below_zero(self):
        with pytest.raises(ValidationError):
            PipelineConfig(min_face_score=-0.1)

    def test_min_face_score_above_one(self):
        with pytest.raises(ValidationError):
            PipelineConfig(min_face_score=1.1)

    def test_min_face_score_boundary_zero(self):
        config = PipelineConfig(min_face_score=0.0)
        assert config.min_face_score == 0.0

    def test_min_face_score_boundary_one(self):
        config = PipelineConfig(min_face_score=1.0)
        assert config.min_face_score == 1.0

    def test_iou_match_thresh_below_zero(self):
        with pytest.raises(ValidationError):
            PipelineConfig(iou_match_thresh=-0.1)

    def test_iou_match_thresh_above_one(self):
        with pytest.raises(ValidationError):
            PipelineConfig(iou_match_thresh=1.5)

    def test_bbox_smoothing_alpha_below_zero(self):
        with pytest.raises(ValidationError):
            PipelineConfig(bbox_smoothing_alpha=-0.1)

    def test_bbox_smoothing_alpha_above_one(self):
        with pytest.raises(ValidationError):
            PipelineConfig(bbox_smoothing_alpha=1.5)

    def test_max_missed_frames_negative(self):
        with pytest.raises(ValidationError):
            PipelineConfig(max_missed_frames=-1)

    def test_crop_padding_negative(self):
        with pytest.raises(ValidationError):
            PipelineConfig(crop_padding=-0.1)

    def test_edge_feather_px_negative(self):
        with pytest.raises(ValidationError):
            PipelineConfig(edge_feather_px=-1)

    def test_max_faces_per_frame_zero(self):
        with pytest.raises(ValidationError):
            PipelineConfig(max_faces_per_frame=0)

    def test_max_faces_per_frame_negative(self):
        with pytest.raises(ValidationError):
            PipelineConfig(max_faces_per_frame=-1)

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            PipelineConfig(device="gpu")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineConfig(unknown_field="value")

