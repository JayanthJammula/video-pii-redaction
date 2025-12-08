"""Tests for face anonymization model wrapper."""

import numpy as np
import pytest

from video_pii.anon.face_anon_wrapper import AnonCode, FaceAnonModel


class TestAnonCode:
    """Tests for AnonCode dataclass."""
    
    def test_construct_with_seed(self):
        code = AnonCode(seed=12345)
        assert code.seed == 12345
    
    def test_latent_defaults_to_none(self):
        code = AnonCode(seed=123)
        assert code.latent is None
    
    def test_construct_with_latent(self):
        latent = np.random.randn(512).astype(np.float32)
        code = AnonCode(seed=123, latent=latent)
        assert code.latent is not None
        assert code.latent.shape == (512,)
    
    def test_hash_based_on_seed(self):
        code1 = AnonCode(seed=123)
        code2 = AnonCode(seed=123)
        code3 = AnonCode(seed=456)
        
        assert hash(code1) == hash(code2)
        assert hash(code1) != hash(code3)
    
    def test_equality(self):
        code1 = AnonCode(seed=123)
        code2 = AnonCode(seed=123)
        code3 = AnonCode(seed=456)
        
        assert code1 == code2
        assert code1 != code3


class TestFaceAnonModelSampleAnonCode:
    """Tests for AnonCode sampling."""
    
    def test_same_track_id_same_code(self):
        model = FaceAnonModel(global_seed=42)
        
        code1 = model.sample_anon_code(track_id=5)
        code2 = model.sample_anon_code(track_id=5)
        
        assert code1 == code2
    
    def test_different_track_ids_different_codes(self):
        model = FaceAnonModel(global_seed=42)
        
        code1 = model.sample_anon_code(track_id=1)
        code2 = model.sample_anon_code(track_id=2)
        code3 = model.sample_anon_code(track_id=3)
        
        assert code1 != code2
        assert code2 != code3
        assert code1 != code3
    
    def test_different_global_seeds_different_codes(self):
        model1 = FaceAnonModel(global_seed=42)
        model2 = FaceAnonModel(global_seed=99)
        
        code1 = model1.sample_anon_code(track_id=5)
        code2 = model2.sample_anon_code(track_id=5)
        
        assert code1 != code2
    
    def test_code_is_deterministic(self):
        # Create two models with same settings
        model1 = FaceAnonModel(global_seed=42)
        model2 = FaceAnonModel(global_seed=42)
        
        # Sample from both
        code1 = model1.sample_anon_code(track_id=10)
        code2 = model2.sample_anon_code(track_id=10)
        
        assert code1 == code2


class TestFaceAnonModelGenerateAnonFace:
    """Tests for face anonymization generation."""
    
    @pytest.fixture
    def sample_face_patch(self):
        """Create a sample face patch for testing."""
        return np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    
    def test_output_same_shape(self, sample_face_patch):
        model = FaceAnonModel()
        code = model.sample_anon_code(track_id=1)
        
        result = model.generate_anon_face(sample_face_patch, code)
        
        assert result.shape == sample_face_patch.shape
    
    def test_output_is_uint8(self, sample_face_patch):
        model = FaceAnonModel()
        code = model.sample_anon_code(track_id=1)
        
        result = model.generate_anon_face(sample_face_patch, code)
        
        assert result.dtype == np.uint8
    
    def test_same_code_same_patch_same_output(self):
        model = FaceAnonModel(global_seed=42)
        face_patch = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        code = model.sample_anon_code(track_id=1)
        
        result1 = model.generate_anon_face(face_patch.copy(), code)
        result2 = model.generate_anon_face(face_patch.copy(), code)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_code_different_output(self):
        model = FaceAnonModel(global_seed=42)
        face_patch = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        code1 = model.sample_anon_code(track_id=1)
        code2 = model.sample_anon_code(track_id=2)
        
        result1 = model.generate_anon_face(face_patch.copy(), code1)
        result2 = model.generate_anon_face(face_patch.copy(), code2)
        
        # Results should differ
        assert not np.array_equal(result1, result2)
    
    def test_empty_patch_returns_empty(self):
        model = FaceAnonModel()
        empty_patch = np.zeros((0, 0, 3), dtype=np.uint8)
        code = model.sample_anon_code(track_id=1)
        
        result = model.generate_anon_face(empty_patch, code)
        
        assert result.shape == empty_patch.shape


class TestFaceAnonModelCustomInference:
    """Tests for custom inference function injection."""
    
    def test_custom_inference_fn_called(self):
        call_count = [0]
        
        def custom_fn(face_patch, anon_code):
            call_count[0] += 1
            return np.zeros_like(face_patch)
        
        model = FaceAnonModel(inference_fn=custom_fn)
        face_patch = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        code = model.sample_anon_code(track_id=1)
        
        model.generate_anon_face(face_patch, code)
        
        assert call_count[0] == 1
    
    def test_custom_inference_receives_correct_args(self):
        received_args = {}
        
        def custom_fn(face_patch, anon_code):
            received_args['face_patch'] = face_patch
            received_args['anon_code'] = anon_code
            return face_patch.copy()
        
        model = FaceAnonModel(inference_fn=custom_fn)
        face_patch = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        code = AnonCode(seed=999)
        
        model.generate_anon_face(face_patch, code)
        
        np.testing.assert_array_equal(received_args['face_patch'], face_patch)
        assert received_args['anon_code'] == code

