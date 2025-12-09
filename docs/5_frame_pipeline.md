# 5-Frame Face Anonymization Pipeline

**Script:** `face_anon_simple/run_5frames.py`

**Output:**
- `output_5frames_face0.mp4` - Video with 5 anonymized frames
- `output_frame_0.png` to `output_frame_4.png` - Individual frames

---

## Part 1: Dependencies & Imports

**File:** `face_anon_simple/run_5frames.py` (lines 1-17)

| Library | Purpose |
|---------|---------|
| `cv2` | Video I/O, image warping |
| `numpy` | Array operations |
| `PIL.Image` | Image format conversion |
| `face_alignment` | Face detection & 68-landmark extraction |
| `torch` | Random seed generator |
| `transformers` | CLIP image encoder |
| `diffusers` | VAE, schedulers |
| `src.diffusers.models.referencenet` | UNet, ReferenceNet models |
| `src.diffusers.pipelines.referencenet` | Anonymization pipeline |
| `utils.extractor` | Face extraction function |

---

## Part 2: Face Detection Model

**File:** `face_anon_simple/run_5frames.py` (lines 22-26)

```python
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, face_detector='sfd', device='cpu'
)
```

| Setting | Value | Purpose |
|---------|-------|---------|
| `LandmarksType` | `TWO_D` | Returns 68 2D facial landmarks |
| `face_detector` | `sfd` | S3FD detector (accurate) |
| `device` | `cpu` | Runs on CPU |

---

## Part 3: Anonymization Model Loading

**File:** `face_anon_simple/run_5frames.py` (lines 28-77)

### 3.1 Model Sources

| Component | Source |
|-----------|--------|
| UNet, ReferenceNets | `hkung/face-anon-simple` |
| VAE, Scheduler | `Charles-Elena/stable-diffusion-2-1` |
| CLIP | `openai/clip-vit-large-patch14` |

### 3.2 Pipeline Assembly

```python
pipe = StableDiffusionReferenceNetPipeline(
    unet=unet,
    referencenet=referencenet,
    conditioning_referencenet=conditioning_referencenet,
    vae=vae,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    scheduler=scheduler,
)
pipe = pipe.to('cpu')
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
```

---

## Part 4: Face Extraction

**File:** `face_anon_simple/utils/extractor.py` (lines 291-314)

### 4.1 Function Signature

```python
def extract_faces(model, image, face_image_size, face_type=FaceType.WHOLE_FACE):
```

### 4.2 Steps

1. **Detect landmarks:** `preds = model.get_landmarks(array)`
2. **Compute affine matrix:** `get_transform_mat(face_landmarks, face_image_size, face_type)`
3. **Warp to 512x512:** `cv2.warpAffine(..., cv2.INTER_LANCZOS4, ...)`

### 4.3 Returns

| Return | Type | Description |
|--------|------|-------------|
| `face_images` | `List[PIL.Image]` | 512x512 aligned face images |
| `image_to_face_matrices` | `List[np.ndarray]` | 2x3 affine matrices |

---

## Part 5: Affine Transform (Umeyama Algorithm)

**File:** `face_anon_simple/utils/extractor.py` (lines 105-173)

Computes similarity transformation (rotation, scale, translation) between source and target landmarks.

### 5.1 Input

- 32 source landmarks (indices 17-48 + 54) from detected face
- 32 target landmarks (normalized 0-1 coordinates in `landmarks_2D_new`)

### 5.2 Output

- 2x3 affine matrix that transforms image coordinates to 512x512 aligned space

---

## Part 6: Anonymization Function

**File:** `face_anon_simple/run_5frames.py` (lines 107-129)

```python
def anonymize_face(face_pil, pipeline, seed=42):
    generator = torch.manual_seed(seed)

    result = pipeline(
        source_image=face_pil,
        conditioning_image=face_pil,
        guidance_scale=7.0,
        num_inference_steps=50,
        anonymization_degree=1.1,
        width=512,
        height=512,
        generator=generator,
    ).images[0]
    return result
```

### 6.1 Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `source_image` | Input face | Face to anonymize |
| `conditioning_image` | Same as source | Reference for structure |
| `guidance_scale` | 7.0 | Classifier-free guidance strength |
| `num_inference_steps` | 50 | Diffusion denoising steps |
| `anonymization_degree` | 1.1 | Identity change amount (>1 = more) |
| `generator` | `torch.manual_seed(42)` | **Fixed seed for consistent identity** |

---

## Part 7: Identity Consistency via Fixed Seed

**File:** `face_anon_simple/run_5frames.py` (lines 143-145, 171-172)

```python
FACE_0_SEED = 42  # Consistent identity for face 0

anon_face = anonymize_face(face_0, pipe, seed=FACE_0_SEED)
```

### 7.1 How It Works

| Step | Without Fixed Seed | With Fixed Seed |
|------|-------------------|-----------------|
| Initial noise | Random each frame | Same each frame |
| Diffusion path | Different | Identical |
| Output identity | Changes per frame | Consistent |

---

## Part 8: Face Paste-Back

**File:** `face_anon_simple/run_5frames.py` (lines 80-104)

### 8.1 Function Signature

```python
def paste_face(fg_pil, bg_pil, mat):
```

### 8.2 Steps

| Step | Code | Purpose |
|------|------|---------|
| **Warp face** | `cv2.warpAffine(fg, mat, ..., WARP_INVERSE_MAP \| INTER_LANCZOS4)` | Transform 512x512 face back to original position |
| **Create mask** | `np.ones((512, 512), dtype=np.uint8) * 255` | White mask for face region |
| **Warp mask** | `cv2.warpAffine(mask, mat, ..., WARP_INVERSE_MAP \| INTER_NEAREST)` | Transform mask to original position |
| **Feather edges** | `cv2.GaussianBlur(warped_mask, (21, 21), 0)` | Smooth mask edges |
| **Blend** | `mask * face + (1 - mask) * background` | Alpha composite |

---

## Part 9: Video I/O

**File:** `face_anon_simple/run_5frames.py` (lines 132-141, 185-192)

### 9.1 Input

```python
cap = cv2.VideoCapture('../test_data/short_clip.mp4')
```

### 9.2 Output

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../output_5frames_face0.mp4', fourcc, fps, (width, height))
```

---

## Part 10: Processing Loop

**File:** `face_anon_simple/run_5frames.py` (lines 147-189)

```
For each frame (1 to 5):
    1. Read frame from video (BGR)
    2. Convert BGR → RGB → PIL
    3. Extract faces → get face images + matrices
    4. Anonymize face 0 with fixed seed
    5. Paste anonymized face back
    6. Convert RGB → BGR
    7. Write to output video
    8. Save as PNG
```

---

## Summary: Data Flow

```
Video Frame (3840x2160 BGR)
    ↓
Convert to RGB PIL Image
    ↓
Face Detection (face_alignment + SFD)
    ↓
68 Landmarks
    ↓
Affine Matrix (umeyama)
    ↓
Warp to 512x512 Aligned Face
    ↓
Anonymize (diffusion model + fixed seed)
    ↓
512x512 Anonymized Face
    ↓
Inverse Warp + Alpha Blend
    ↓
Output Frame (3840x2160 BGR)
```

