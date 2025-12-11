#!/usr/bin/env python
"""
Face Anonymization Pipeline

Anonymizes faces in video while preserving consistent identity across frames.
Uses ReferenceNet-based diffusion model for realistic face generation.

Usage:
    cd face_anon_simple
    python run_5frames.py --input ../test_data/short_clip.mp4 --output ../output.mp4 --num_frames 5
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import face_alignment
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler

from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
from utils.extractor import get_transform_mat, FaceType
from utils.tracker import FaceTracker


# ============================================================================
# Configuration
# ============================================================================

# Model paths (use local_files_only=True since models are cached)
FACE_MODEL_ID = "hkung/face-anon-simple"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
SD_MODEL_ID = "Charles-Elena/stable-diffusion-2-1"

# Anonymization parameters
GUIDANCE_SCALE = 7.0        # CFG scale (higher = more guidance)
NUM_INFERENCE_STEPS = 75    # Diffusion steps (more = better quality, slower)
ANONYMIZATION_DEGREE = 0.1  # How different from original (1.0 = same, higher = more different)
FACE_SIZE = 512             # Face crop size for model input

# Identity seeds - same seed = same anonymized identity across all frames
# Use different seeds for different tracked people
DEFAULT_SEED = 42


# ============================================================================
# Model Loading
# ============================================================================

def load_models(device='cpu'):
    """Load all model components."""
    print('Loading face alignment model...')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector='sfd', device='cpu'
    )

    print('Loading anonymization model components...')
    print(f'  Device: {device}')

    print('  Loading UNet...')
    unet = UNet2DConditionModel.from_pretrained(
        FACE_MODEL_ID, subfolder="unet", use_safetensors=True, local_files_only=True
    )
    print('  Loading ReferenceNet...')
    referencenet = ReferenceNetModel.from_pretrained(
        FACE_MODEL_ID, subfolder="referencenet", use_safetensors=True, local_files_only=True
    )
    print('  Loading Conditioning ReferenceNet...')
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        FACE_MODEL_ID, subfolder="conditioning_referencenet", use_safetensors=True, local_files_only=True
    )
    print('  Loading VAE...')
    vae = AutoencoderKL.from_pretrained(
        SD_MODEL_ID, subfolder="vae", use_safetensors=True, local_files_only=True
    )
    print('  Loading Scheduler...')
    scheduler = DDPMScheduler.from_pretrained(
        SD_MODEL_ID, subfolder="scheduler", local_files_only=True
    )
    print('  Loading CLIP...')
    feature_extractor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
    image_encoder = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)

    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    print('Models loaded!')
    return fa, pipe


# ============================================================================
# Face Processing Functions
# ============================================================================

def paste_face(fg_pil, bg_pil, mat):
    """Paste anonymized face back onto original frame.

    Uses inverse affine transform with Lanczos interpolation and
    Gaussian-blurred mask for seamless blending.

    Args:
        fg_pil: PIL Image of anonymized face (512x512)
        bg_pil: PIL Image of original frame
        mat: 2x3 affine transformation matrix from extraction

    Returns:
        PIL Image with face blended back
    """
    fg = np.array(fg_pil)
    bg = np.array(bg_pil).copy()
    h, w = bg.shape[:2]

    # Warp face back to original position
    warped = cv2.warpAffine(fg, mat, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Create and warp mask
    mask = np.ones((fg.shape[0], fg.shape[1]), dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(mask, mat, (w, h),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Feather edges for seamless blending
    warped_mask = cv2.GaussianBlur(warped_mask, (21, 21), 0)
    mask_f = warped_mask.astype(np.float32) / 255.0
    mask_3 = np.stack([mask_f]*3, axis=-1)

    # Alpha blend
    result = (mask_3 * warped + (1 - mask_3) * bg).astype(np.uint8)
    return Image.fromarray(result)


def load_reference_image(path):
    """Load and resize custom reference image to the expected face size."""
    image = Image.open(path).convert('RGB')
    if image.size != (FACE_SIZE, FACE_SIZE):
        image = image.resize((FACE_SIZE, FACE_SIZE), Image.LANCZOS)
    return image


def anonymize_face(face_pil, pipeline, seed=DEFAULT_SEED, reference_image=None):
    """Anonymize a single face with consistent identity.

    Uses fixed seed to ensure the same person gets the same
    anonymized identity across all frames.

    Args:
        face_pil: PIL Image of the face (512x512)
        pipeline: The anonymization pipeline
        seed: Random seed for identity consistency (same seed = same identity)

    Returns:
        PIL Image of anonymized face (512x512)
    """
    generator = torch.manual_seed(seed)

    source_image = reference_image if reference_image is not None else face_pil

    result = pipeline(
        source_image=source_image,
        conditioning_image=face_pil,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        anonymization_degree=ANONYMIZATION_DEGREE,
        width=FACE_SIZE,
        height=FACE_SIZE,
        generator=generator,
    ).images[0]
    return result


# ============================================================================
# Video Processing
# ============================================================================

def process_video(input_path, output_path, fa, pipeline, num_frames=None,
                  base_seed=DEFAULT_SEED, save_frames=False, reference_image=None):
    """Process video and anonymize ALL faces with tracking.

    Uses IoU-based tracking to maintain consistent identity for each person
    across frames. Each tracked person gets a unique seed derived from their
    track ID.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        fa: Face alignment model
        pipeline: Anonymization pipeline
        num_frames: Number of frames to process (None = all)
        base_seed: Base seed for identity consistency (each face gets base_seed + track_id * 1000)
        save_frames: Whether to save individual frames as PNG
        reference_image: Optional PIL image used as the source identity
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames is None:
        num_frames = total_frames

    print(f'Input: {input_path}')
    print(f'Video: {width}x{height} @ {fps}fps ({total_frames} frames)')
    print(f'Processing: {num_frames} frames')
    print()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize face tracker
    tracker = FaceTracker(iou_threshold=0.3, max_lost_frames=30)

    for frame_idx in range(num_frames):
        print(f'Processing frame {frame_idx + 1}/{num_frames}...')

        ret, frame_bgr = cap.read()
        if not ret:
            print('  No more frames in video!')
            break

        # Convert to PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_array = np.array(frame_rgb)
        pil_frame = Image.fromarray(frame_rgb)

        # Detect face landmarks
        landmarks_list = fa.get_landmarks(frame_array)

        if landmarks_list is None or len(landmarks_list) == 0:
            print('  No faces detected')
            out.write(frame_bgr)
            if save_frames:
                frame_path = output_path.replace('.mp4', f'_frame_{frame_idx}.png')
                cv2.imwrite(frame_path, frame_bgr)
            tracker.update([])  # Update tracker with no detections
            continue

        # Update tracker with current detections
        track_ids = tracker.update(landmarks_list)
        print(f'  Found {len(landmarks_list)} face(s), track IDs: {track_ids}')

        # Process only the first face (largest/most prominent)
        result_pil = pil_frame
        landmarks = landmarks_list[0]
        track_id = track_ids[0]
        face_idx = 0
        if True:  # Single face block (keeps indentation consistent for easy revert)
            # Extract face using landmarks
            mat = get_transform_mat(landmarks, FACE_SIZE, FaceType.WHOLE_FACE)
            face_array = cv2.warpAffine(
                frame_array, mat, (FACE_SIZE, FACE_SIZE),
                cv2.INTER_LANCZOS4, borderValue=(255, 255, 255)
            )
            face_pil = Image.fromarray(face_array)

            # Get unique seed for this tracked person
            face_seed = tracker.get_seed_for_track(track_id, base_seed)
            print(f'  Face {face_idx}: track_id={track_id}, seed={face_seed}')

            # Anonymize
            anon_face = anonymize_face(
                face_pil,
                pipeline,
                seed=face_seed,
                reference_image=reference_image,
            )

            # Paste back
            result_pil = paste_face(anon_face, result_pil, mat)

        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        out.write(result_bgr)

        if save_frames:
            frame_path = output_path.replace('.mp4', f'_frame_{frame_idx}.png')
            cv2.imwrite(frame_path, result_bgr)

    cap.release()
    out.release()
    print()
    print(f'Done! Output saved to: {output_path}')
    print(f'Total unique faces tracked: {tracker.next_id}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Anonymize faces in video')
    parser.add_argument('--input', '-i', default='../test_data/short_clip.mp4',
                        help='Input video path')
    parser.add_argument('--output', '-o', default='../output_anonymized.mp4',
                        help='Output video path')
    parser.add_argument('--num_frames', '-n', type=int, default=None,
                        help='Number of frames to process (default: all)')
    parser.add_argument('--seed', '-s', type=int, default=DEFAULT_SEED,
                        help='Base seed for identity consistency (each face gets unique seed)')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save individual frames as PNG')
    parser.add_argument('--device', '-d', default='cpu',
                        help='Device to use (cpu/cuda/mps)')
    parser.add_argument('--reference_image', '-r', default=None,
                        help='Optional path to an image used as the source identity (e.g., 1.png)')
    args = parser.parse_args()

    print('=== Face Anonymization Pipeline (with Tracking) ===')
    print()

    # Load models
    fa, pipeline = load_models(device=args.device)
    reference_image = None
    if args.reference_image:
        print(f'Loading reference image from {args.reference_image}...')
        reference_image = load_reference_image(args.reference_image)

    # Process video
    process_video(
        input_path=args.input,
        output_path=args.output,
        fa=fa,
        pipeline=pipeline,
        num_frames=args.num_frames,
        base_seed=args.seed,
        save_frames=args.save_frames,
        reference_image=reference_image,
    )


if __name__ == '__main__':
    main()
