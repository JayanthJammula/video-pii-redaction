#!/usr/bin/env python3
"""CLI script for video face anonymization."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_pii.config import PipelineConfig
from video_pii.pipeline.video_pipeline import run_video_anonymization
from video_pii.utils.device import get_best_device, list_available_devices


def print_device_info() -> None:
    """Print information about available compute devices."""
    devices = list_available_devices()
    print("Available compute devices:")
    print("-" * 40)
    for dev in devices:
        mem_str = f" ({dev.memory_gb:.1f} GB)" if dev.memory_gb else ""
        print(f"  {dev.name:<12} [{dev.type}]{mem_str}")
    print("-" * 40)
    best = get_best_device()
    print(f"Best available: {best}")
    print()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anonymize faces in a video using SCRFD detection and diffusion-based face anonymization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Device options
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available compute devices and exit.",
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input video file.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for output video file.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cuda:0, cpu, mps). 'auto' selects best available.",
    )
    
    parser.add_argument(
        "--min-face-score",
        type=float,
        default=0.4,
        help="Minimum face detection confidence score.",
    )
    
    parser.add_argument(
        "--iou-match-thresh",
        type=float,
        default=0.4,
        help="IoU threshold for matching detections to tracks.",
    )
    
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=10,
        help="Max frames a track can be missed before removal.",
    )
    
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.3,
        help="Padding ratio around face bbox for cropping.",
    )
    
    parser.add_argument(
        "--bbox-smoothing-alpha",
        type=float,
        default=0.7,
        help="Smoothing factor for bbox EMA (0-1).",
    )
    
    parser.add_argument(
        "--edge-feather-px",
        type=int,
        default=8,
        help="Pixels to feather at edges when pasting.",
    )
    
    parser.add_argument(
        "--max-faces-per-frame",
        type=int,
        default=16,
        help="Maximum faces to process per frame.",
    )
    
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw debug overlays on output.",
    )

    # Model configuration
    parser.add_argument(
        "--scrfd-model",
        type=str,
        default="buffalo_l",
        help="InsightFace model name for SCRFD (e.g., 'buffalo_l', 'buffalo_sc').",
    )

    parser.add_argument(
        "--anon-model",
        type=str,
        default=None,
        help="HuggingFace model ID for face anonymization (default: mock mode).",
    )

    parser.add_argument(
        "--anon-steps",
        type=int,
        default=25,
        help="Number of diffusion inference steps (lower = faster).",
    )

    parser.add_argument(
        "--anon-guidance",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale.",
    )

    parser.add_argument(
        "--anon-degree",
        type=float,
        default=1.25,
        help="Anonymization degree (higher = more different from original).",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args(args)


def create_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Create PipelineConfig from parsed arguments."""
    return PipelineConfig(
        device=args.device,
        min_face_score=args.min_face_score,
        iou_match_thresh=args.iou_match_thresh,
        max_missed_frames=args.max_missed_frames,
        crop_padding=args.crop_padding,
        bbox_smoothing_alpha=args.bbox_smoothing_alpha,
        edge_feather_px=args.edge_feather_px,
        max_faces_per_frame=args.max_faces_per_frame,
        debug_overlay=args.debug_overlay,
        # Model configuration
        scrfd_model_name=args.scrfd_model,
        anon_model_id=args.anon_model,
        anon_num_steps=args.anon_steps,
        anon_guidance_scale=args.anon_guidance,
        anon_degree=args.anon_degree,
    )


def main(args: list[str] | None = None) -> int:
    """Main entry point."""
    parsed = parse_args(args)

    # Handle --list-devices
    if parsed.list_devices:
        print_device_info()
        return 0

    # Validate required args
    if not parsed.input:
        print("Error: --input/-i is required", file=sys.stderr)
        return 1
    if not parsed.output:
        print("Error: --output/-o is required", file=sys.stderr)
        return 1

    # Validate input exists
    if not Path(parsed.input).exists():
        print(f"Error: Input file not found: {parsed.input}", file=sys.stderr)
        return 1

    # Create config
    config = create_config_from_args(parsed)

    # Print device info
    if not parsed.quiet:
        print(f"Using device: {config.device}")
    
    # Progress callback
    def progress(frame_idx: int, total: int) -> None:
        if not parsed.quiet:
            if total > 0:
                pct = (frame_idx + 1) / total * 100
                print(f"\rProcessing: {frame_idx + 1}/{total} ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\rProcessing frame {frame_idx + 1}...", end="", flush=True)
    
    # Run pipeline
    try:
        result = run_video_anonymization(
            input_path=parsed.input,
            output_path=parsed.output,
            config=config,
            progress_callback=progress if not parsed.quiet else None,
        )
        
        if not parsed.quiet:
            print()  # Newline after progress
            print(f"Done! Processed {result['frames_processed']} frames.")
            print(f"Output saved to: {result['output_path']}")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

