"""Video pipeline for full video anonymization."""

from typing import Callable

from video_pii.anon.face_anon_wrapper import FaceAnonModel
from video_pii.config import PipelineConfig
from video_pii.detect.scrfd_detector import SCRFDFaceDetector
from video_pii.io.video_io import VideoWriter, probe_video, read_video_frames
from video_pii.pipeline.frame_processor import FrameProcessor
from video_pii.track.face_tracker import FaceTracker


# Type aliases for factory functions (for dependency injection in tests)
DetectorFactory = Callable[[PipelineConfig], SCRFDFaceDetector]
TrackerFactory = Callable[[PipelineConfig], FaceTracker]
AnonModelFactory = Callable[[PipelineConfig], FaceAnonModel]


def default_detector_factory(config: PipelineConfig) -> SCRFDFaceDetector:
    """Create default SCRFD detector using insightface.

    Uses 'buffalo_l' model by default for best accuracy.
    """
    return SCRFDFaceDetector(
        model_name=config.scrfd_model_name,
        device=config.device,
        min_face_score=config.min_face_score,
    )


def default_tracker_factory(config: PipelineConfig) -> FaceTracker:
    """Create default face tracker."""
    return FaceTracker(
        iou_match_thresh=config.iou_match_thresh,
        max_missed_frames=config.max_missed_frames,
        bbox_smoothing_alpha=config.bbox_smoothing_alpha,
    )


def default_anon_model_factory(config: PipelineConfig) -> FaceAnonModel:
    """Create default face anonymization model.

    Uses face_anon_simple from HuggingFace if available.
    """
    return FaceAnonModel(
        model_id=config.anon_model_id,
        device=config.device,
        num_inference_steps=config.anon_num_steps,
        guidance_scale=config.anon_guidance_scale,
        anonymization_degree=config.anon_degree,
    )


def run_video_anonymization(
    input_path: str,
    output_path: str,
    config: PipelineConfig | None = None,
    detector_factory: DetectorFactory | None = None,
    tracker_factory: TrackerFactory | None = None,
    anon_model_factory: AnonModelFactory | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run face anonymization on a video file.
    
    Args:
        input_path: Path to input video file.
        output_path: Path for output video file.
        config: Pipeline configuration (uses defaults if None).
        detector_factory: Optional factory for detector (for testing).
        tracker_factory: Optional factory for tracker (for testing).
        anon_model_factory: Optional factory for anon model (for testing).
        progress_callback: Optional callback(frame_idx, total_frames).
        
    Returns:
        Dict with statistics: {'frames_processed': int, 'input_path': str, ...}
    """
    if config is None:
        config = PipelineConfig()
    
    # Use provided factories or defaults
    detector_factory = detector_factory or default_detector_factory
    tracker_factory = tracker_factory or default_tracker_factory
    anon_model_factory = anon_model_factory or default_anon_model_factory
    
    # Probe input video
    meta = probe_video(input_path)
    
    # Create components
    detector = detector_factory(config)
    tracker = tracker_factory(config)
    anon_model = anon_model_factory(config)
    
    # Create frame processor
    processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        anon_model=anon_model,
        config=config,
    )
    
    # Process video
    frames_processed = 0
    total_frames = meta.frame_count or 0
    
    with VideoWriter(output_path, meta) as writer:
        for frame_idx, frame in enumerate(read_video_frames(input_path)):
            # Process frame
            processed = processor.process_frame(frame, frame_idx)
            
            # Write frame
            writer.write_frame(processed)
            
            frames_processed += 1
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(frame_idx, total_frames)
    
    return {
        "input_path": input_path,
        "output_path": output_path,
        "frames_processed": frames_processed,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
    }

