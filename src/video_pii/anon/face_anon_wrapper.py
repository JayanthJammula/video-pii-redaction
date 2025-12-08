"""Face anonymization model wrapper with consistent identity support.

Integrates with face_anon_simple (WACV 2025) for diffusion-based anonymization.
"""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass
class AnonCode:
    """Anonymization code for consistent identity.

    Stores a deterministic seed derived from track_id, ensuring the same
    real person always maps to the same anonymized appearance.

    Attributes:
        seed: Integer seed for deterministic generation.
        latent: Optional latent vector for more complex identity encoding.
    """
    seed: int
    latent: np.ndarray | None = None

    def __hash__(self) -> int:
        return hash(self.seed)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnonCode):
            return False
        return self.seed == other.seed


class AnonInferenceFn(Protocol):
    """Protocol for anonymization inference function."""

    def __call__(
        self,
        face_patch: np.ndarray,
        anon_code: AnonCode
    ) -> np.ndarray:
        """Generate anonymized face.

        Args:
            face_patch: Face image (H, W, 3) in BGR or RGB.
            anon_code: Anonymization code for identity.

        Returns:
            Anonymized face image with same shape.
        """
        ...


def default_mock_inference(face_patch: np.ndarray, anon_code: AnonCode) -> np.ndarray:
    """Default mock inference for testing.
    
    Applies a deterministic color shift based on the anon code.
    """
    rng = np.random.default_rng(seed=anon_code.seed)
    
    # Generate a color shift based on the seed
    color_shift = rng.integers(0, 256, size=3, dtype=np.uint8)
    
    # Apply XOR with color shift for deterministic transformation
    result = face_patch.copy()
    for c in range(3):
        result[:, :, c] = (result[:, :, c].astype(np.int32) + color_shift[c]) % 256
    
    return result.astype(np.uint8)


class FaceAnonModel:
    """Wrapper for face anonymization model with consistent identity.

    Supports dependency injection of inference function for testing.
    In production, integrates with face_anon_simple diffusion model.
    """

    # Default model IDs from HuggingFace
    FACE_MODEL_ID = "hkung/face-anon-simple"
    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    # Use community mirror of SD 2.1 (original stabilityai repo was removed)
    # This contains the correct VAE that the face_anon_simple model was trained with
    SD_MODEL_ID = "Charles-Elena/stable-diffusion-2-1"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        global_seed: int = 42,
        inference_fn: AnonInferenceFn | None = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 4.0,
        anonymization_degree: float = 1.25,
    ):
        """Initialize the face anonymization model.

        Args:
            model_id: HuggingFace model ID (defaults to 'hkung/face-anon-simple').
            device: Compute device ('cuda', 'cpu', 'mps').
            global_seed: Global seed for reproducibility.
            inference_fn: Optional inference function for testing.
            num_inference_steps: Number of diffusion steps (lower = faster).
            guidance_scale: Classifier-free guidance scale.
            anonymization_degree: Degree of anonymization (higher = more different).
        """
        self.device = device
        self.global_seed = global_seed
        self._inference_fn = inference_fn
        self._pipe = None
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._anonymization_degree = anonymization_degree

        if inference_fn is None and model_id is not None:
            self._load_model(model_id)

    def _load_model(self, model_id: str) -> None:
        """Load the face_anon_simple pipeline from HuggingFace.

        Args:
            model_id: HuggingFace model ID for face_anon_simple.
        """
        import sys
        import os

        # Add face_anon_simple repo to path if it exists locally
        repo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "face_anon_simple")
        if os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            import torch
            from transformers import CLIPImageProcessor, CLIPVisionModel
            from diffusers import AutoencoderKL, DDPMScheduler

            # Try importing from face_anon_simple's custom modules
            try:
                from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
                    ReferenceNetModel,
                )
                from src.diffusers.models.referencenet.unet_2d_condition import (
                    UNet2DConditionModel,
                )
                from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
                    StableDiffusionReferenceNetPipeline,
                )
            except ImportError:
                import warnings
                warnings.warn(
                    "face_anon_simple custom modules not found. "
                    "Clone https://github.com/hanweikung/face_anon_simple "
                    "into the project root directory."
                )
                return

            # Load model components
            unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", use_safetensors=True
            )
            referencenet = ReferenceNetModel.from_pretrained(
                model_id, subfolder="referencenet", use_safetensors=True
            )
            conditioning_referencenet = ReferenceNetModel.from_pretrained(
                model_id, subfolder="conditioning_referencenet", use_safetensors=True
            )
            # Load VAE from SD 2.1 community mirror (original stabilityai repo removed)
            # This is the correct VAE that face_anon_simple was trained with
            vae = AutoencoderKL.from_pretrained(
                self.SD_MODEL_ID, subfolder="vae", use_safetensors=True
            )

            # Load scheduler from SD 2.1 community mirror
            scheduler = DDPMScheduler.from_pretrained(
                self.SD_MODEL_ID, subfolder="scheduler"
            )
            feature_extractor = CLIPImageProcessor.from_pretrained(
                self.CLIP_MODEL_ID
            )
            image_encoder = CLIPVisionModel.from_pretrained(
                self.CLIP_MODEL_ID
            )

            # Create pipeline
            self._pipe = StableDiffusionReferenceNetPipeline(
                unet=unet,
                referencenet=referencenet,
                conditioning_referencenet=conditioning_referencenet,
                vae=vae,
                feature_extractor=feature_extractor,
                image_encoder=image_encoder,
                scheduler=scheduler,
            )
            self._pipe = self._pipe.to(self.device)

        except ImportError as e:
            import warnings
            warnings.warn(
                f"Failed to load face_anon_simple model: {e}. "
                "Install dependencies: pip install diffusers transformers"
            )
            self._pipe = None
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load face_anon_simple model: {e}")
            self._pipe = None

    def sample_anon_code(self, track_id: int) -> AnonCode:
        """Generate a deterministic anonymization code for a track.

        The same track_id always produces the same AnonCode, ensuring
        consistent anonymized appearance across frames.

        Args:
            track_id: Unique identifier for the tracked face.

        Returns:
            AnonCode with deterministic seed.
        """
        # Combine global seed with track_id for deterministic but unique codes
        combined_seed = (self.global_seed * 1000000 + track_id) % (2**31)
        return AnonCode(seed=combined_seed)

    def generate_anon_face(
        self,
        face_patch: np.ndarray,
        anon_code: AnonCode
    ) -> np.ndarray:
        """Generate an anonymized version of the face.

        Args:
            face_patch: Face image (H, W, 3) in BGR format.
            anon_code: Anonymization code for consistent identity.

        Returns:
            Anonymized face image with same shape and format.
        """
        if face_patch.size == 0:
            return face_patch.copy()

        if self._inference_fn is not None:
            return self._inference_fn(face_patch, anon_code)

        if self._pipe is not None:
            return self._run_model(face_patch, anon_code)

        # No inference fn or model - use mock for testing
        return default_mock_inference(face_patch, anon_code)

    def _run_model(
        self,
        face_patch: np.ndarray,
        anon_code: AnonCode
    ) -> np.ndarray:
        """Run the face_anon_simple diffusion pipeline.

        Args:
            face_patch: Face image (H, W, 3) in BGR format.
            anon_code: Anonymization code with seed for generator.

        Returns:
            Anonymized face image with same shape and format.
        """
        import torch
        from PIL import Image

        original_h, original_w = face_patch.shape[:2]

        # Convert BGR to RGB and to PIL Image
        rgb_patch = cv2.cvtColor(face_patch, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_patch)

        # Create deterministic generator from anon_code
        generator = torch.manual_seed(anon_code.seed)

        # Run the pipeline
        # For anonymization, source and conditioning are the same image
        result = self._pipe(
            source_image=pil_image,
            conditioning_image=pil_image,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            generator=generator,
            anonymization_degree=self._anonymization_degree,
            width=512,
            height=512,
        )

        # Get output image
        anon_pil = result.images[0]

        # Convert back to numpy BGR
        anon_rgb = np.array(anon_pil)
        anon_bgr = cv2.cvtColor(anon_rgb, cv2.COLOR_RGB2BGR)

        # Resize back to original size
        if anon_bgr.shape[:2] != (original_h, original_w):
            anon_bgr = cv2.resize(anon_bgr, (original_w, original_h))

        return anon_bgr

