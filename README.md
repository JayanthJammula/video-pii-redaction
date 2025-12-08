# Video PII Redaction Pipeline

A comprehensive video face anonymization pipeline using SCRFD for detection and diffusion-based face anonymization (face_anon_simple from WACV 2025).

## Features

- **SCRFD Face Detection**: High-accuracy face detection via InsightFace
- **Diffusion-Based Anonymization**: Uses face_anon_simple for realistic face replacement
- **Temporal Tracking**: IoU-based tracking with EMA smoothing for jitter reduction
- **Consistent Identity**: Same tracked face → same anonymized appearance across frames
- **Edge Feathering**: Seamless blending of anonymized faces
- **GPU Support**: CUDA, MPS (Apple Silicon), and CPU backends
- **Auto Device Selection**: Automatically selects the best available compute device

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd PII_Redaction
```

### 2. Install base package

```bash
pip install -e .
```

### 3. Install dependencies for face anonymization

```bash
# Clone face_anon_simple (required for diffusion model)
git clone https://github.com/hanweikung/face_anon_simple.git

# Install with specific versions for compatibility
pip install insightface onnxruntime  # For SCRFD detection
pip install diffusers==0.25.1 transformers==4.46.1 accelerate==1.0.1
pip install "huggingface_hub<0.26.0" "peft<0.14.0"
```

### 4. For GPU support (NVIDIA CUDA)

```bash
pip install onnxruntime-gpu  # Replace onnxruntime for CUDA support
```

## Quick Start

### List available devices

```bash
python scripts/anonymize_video.py --list-devices
```

### Run with auto device selection (recommended)

```bash
python scripts/anonymize_video.py -i input.mp4 -o output.mp4
```

### Run with specific device

```bash
# CPU
python scripts/anonymize_video.py -i input.mp4 -o output.mp4 --device cpu

# CUDA GPU
python scripts/anonymize_video.py -i input.mp4 -o output.mp4 --device cuda

# Specific CUDA device
python scripts/anonymize_video.py -i input.mp4 -o output.mp4 --device cuda:1

# Apple Silicon (MPS)
python scripts/anonymize_video.py -i input.mp4 -o output.mp4 --device mps
```

### Run with real face anonymization model

```bash
# Set PYTHONPATH to include face_anon_simple
export PYTHONPATH=face_anon_simple:$PYTHONPATH

# Run with the diffusion model
python scripts/anonymize_video.py \
    -i input.mp4 \
    -o output.mp4 \
    --anon-model hkung/face-anon-simple \
    --anon-steps 25 \
    --device auto
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | required | Input video file |
| `-o, --output` | required | Output video file |
| `--device` | auto | Compute device (auto, cuda, cuda:N, mps, cpu) |
| `--list-devices` | - | List available devices and exit |
| `--min-face-score` | 0.4 | Minimum face detection confidence |
| `--anon-model` | None | HuggingFace model ID (None = mock mode) |
| `--anon-steps` | 25 | Diffusion inference steps (lower = faster) |
| `--anon-guidance` | 4.0 | Classifier-free guidance scale |
| `--anon-degree` | 1.25 | Anonymization degree |
| `--debug-overlay` | False | Draw debug bounding boxes |
| `-q, --quiet` | False | Suppress progress output |

## GPU Support

### NVIDIA CUDA

For CUDA support, install `onnxruntime-gpu`:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

The pipeline will automatically detect CUDA availability and use it.

### Apple Silicon (MPS)

MPS is automatically detected on Apple Silicon Macs. Note that ONNX Runtime doesn't support MPS directly, so face detection runs on CPU while face anonymization uses MPS.

## Project Structure

```
src/video_pii/
├── config.py                    # PipelineConfig (Pydantic)
├── anon/
│   └── face_anon_wrapper.py     # FaceAnonModel, AnonCode
├── detect/
│   └── scrfd_detector.py        # SCRFDFaceDetector
├── io/
│   └── video_io.py              # VideoMeta, probe_video, VideoWriter
├── pipeline/
│   ├── frame_processor.py       # FrameProcessor
│   └── video_pipeline.py        # run_video_anonymization()
├── track/
│   └── face_tracker.py          # FaceTracker, TrackState
└── utils/
    ├── device.py                # GPU detection utilities
    ├── geometry.py              # iou(), smooth_bbox(), expand_bbox()
    └── visual_debug.py          # Debug visualization

scripts/
└── anonymize_video.py           # CLI entry point
```

## Dependency Versions

The face_anon_simple model requires specific dependency versions:

| Package | Version | Notes |
|---------|---------|-------|
| diffusers | 0.25.1 | Required for face_anon_simple |
| transformers | 4.46.1 | Compatible with diffusers 0.25.1 |
| huggingface_hub | <0.26.0 | For cached_download compatibility |
| peft | <0.14.0 | Avoids transformers.modeling_layers issue |
| accelerate | 1.0.1 | For model acceleration |
| insightface | latest | SCRFD face detection |

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest -v tests/test_config.py
```

## License

MIT

