# Face Anonymization Configuration Guide

## Optimal Configuration

After extensive testing, the following configuration produces the best face anonymization results:

| Parameter | Optimal Value | Description |
|-----------|---------------|-------------|
| `anonymization_degree` | **1.1** | Lower values preserve more facial details (expression, emotion) |
| `guidance_scale` | **7.0** | Higher values preserve structure and expression better |
| `num_inference_steps` | **50** | More steps = sharper, higher quality output |

## Key Technical Findings

### 1. VAE Requirement

The face_anon_simple model was trained with Stable Diffusion 2.1's VAE. The original `stabilityai/stable-diffusion-2-1` repository was removed from HuggingFace, so we use a community mirror:

```python
vae = AutoencoderKL.from_pretrained(
    "Charles-Elena/stable-diffusion-2-1",
    subfolder="vae",
    use_safetensors=True
)
```

**Important:** Using the wrong VAE produces colorful noise/static patterns instead of realistic faces.

### 2. Face Alignment is Critical

The diffusion model expects properly aligned 512x512 face images. We use the `face_alignment` library with the `extract_faces()` utility from face_anon_simple:

```python
import face_alignment
from utils.extractor import extract_faces

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    face_detector='sfd',
    device='cpu'  # or 'cuda' for GPU
)
face_images, matrices = extract_faces(fa, pil_image, 512)
```

**What it does:**
- Detects 68 facial landmarks
- Computes affine transformation using umeyama algorithm
- Warps faces to normalized 512x512 aligned images
- Returns transformation matrices for paste-back

### 3. High-Quality Paste-Back

When pasting the anonymized face back into the original frame, use:

- **`cv2.INTER_LANCZOS4`** interpolation (not `INTER_NEAREST` which causes pixelation)
- **Feathered edges** with Gaussian blur for smooth blending

```python
def paste_foreground_hq(fg_image, bg_image, rotation_matrix):
    fg_array = np.array(fg_image)
    bg_array = np.array(bg_image)[:, :, :3].copy()
    height, width = bg_array.shape[:2]
    
    # Use LANCZOS4 for high-quality upscaling
    warped_fg_255 = cv2.warpAffine(
        fg_array, rotation_matrix, (width, height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    warped_fg_0 = cv2.warpAffine(
        fg_array, rotation_matrix, (width, height),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Create mask and feather edges
    diff = cv2.absdiff(warped_fg_255, warped_fg_0)
    mask = diff / 255.0
    mask_gray = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mask_blurred = cv2.GaussianBlur(mask_gray, (7, 7), 0)
    mask_feathered = np.stack([mask_blurred / 255.0] * 3, axis=-1)
    
    # Blend: bg where mask=1, face where mask=0
    result = mask_feathered * bg_array + warped_fg_0
    return Image.fromarray(result.astype('uint8'), 'RGB')
```

## Configuration in Code

### PipelineConfig (src/video_pii/config.py)

```python
from video_pii.config import PipelineConfig

config = PipelineConfig(
    anon_model_id="hkung/face-anon-simple",
    anon_num_steps=50,        # Optimal
    anon_guidance_scale=7.0,  # Optimal
    anon_degree=1.1,          # Optimal
    device="auto",            # Auto-detect GPU/CPU
)
```

### FaceAnonModel (src/video_pii/anon/face_anon_wrapper.py)

```python
from video_pii.anon.face_anon_wrapper import FaceAnonModel

model = FaceAnonModel(
    model_id="hkung/face-anon-simple",
    device="cuda",  # or "mps" for Apple Silicon, "cpu" for CPU
    num_inference_steps=50,
    guidance_scale=7.0,
    anonymization_degree=1.1,
)
```

## Dependencies

```bash
pip install face_alignment
pip install diffusers transformers
pip install peft==0.13.2  # Must pin this version for compatibility
```

## Performance Notes

- **CPU:** ~5-6 seconds per inference step (50 steps ≈ 5 minutes per face)
- **GPU (CUDA):** Much faster, recommended for video processing
- **Apple Silicon (MPS):** May encounter memory issues with 4K frames; consider downscaling

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Colorful noise output | Wrong VAE | Use `Charles-Elena/stable-diffusion-2-1` VAE |
| Blurry faces | Too few steps or high anon_degree | Use 50 steps, degree=1.1 |
| Expression not preserved | guidance_scale too low | Use guidance_scale=7.0 |
| Pixelated paste-back | INTER_NEAREST interpolation | Use INTER_LANCZOS4 |
| Visible seams | No edge feathering | Apply Gaussian blur to mask |

