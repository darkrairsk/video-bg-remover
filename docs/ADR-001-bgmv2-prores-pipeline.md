# ADR-001: BackgroundMattingV2 + ProRes 4444 Alpha Pipeline

| Field        | Value                                   |
|--------------|-----------------------------------------|
| **Status**   | Accepted                                |
| **Date**     | 2026-04-07                              |
| **Author**   | Rahul                                   |
| **Context**  | Video background removal for selfie clips |

---

## Context

The goal is to remove the background from selfie `.mov` videos recorded against a plain wall (no green screen), and export a file that:

1. Has a **true alpha channel** (transparent background) — not composited onto a colour.
2. Preserves the **original audio** from the input file.
3. Is **natively compatible with CapCut and YouTube Shorts** editing workflows.
4. Runs **fast on Apple Silicon (M2)** without requiring a cloud GPU.

The previous implementation used **MediaPipe Selfie Segmenter** (a TFLite-based segmentation model), which had several limitations:
- It produces only a hard binary mask (no soft alpha/feathering).
- It has no concept of a background plate — segmentation is purely foreground/background classification.
- The output was composited onto a solid colour or image, not a true alpha `.mov`.
- Required approx. 15+ packages including TensorFlow, `absl-py`, `sounddevice`, `matplotlib`, etc.

---

## Decision

### 1. AI Model — BackgroundMattingV2 (BGMv2)

**Chosen:** `BackgroundMattingV2` by Peter Lin et al. (CVPR 2021)  
**Variant:** TorchScript MobileNetV2 `float32` (`torchscript_mobilenetv2_fp32.pth`)

**Why BGMv2 over alternatives:**

| Model | Alpha Quality | Speed (M2) | Background Plate Required | Notes |
|---|---|---|---|---|
| MediaPipe Selfie | Low (binary mask) | Fast | No | No soft edges |
| **BGMv2 (chosen)** | **High (soft matte)** | **Good** | **Yes** | Best quality/speed balance |
| Robust Video Matting (RVM) | High | Good | No | Considered for future upgrade |
| rembg / U²-Net | Medium | Slow | No | No temporal consistency |

BGMv2 produces a soft alpha matte with accurate edge feathering, which is critical for hair and fine details in selfie videos.

**TorchScript variant** was chosen (over the PyTorch research variant) because:
- It bundles architecture + weights into a single `.pth` file.
- No need to clone the BGMv2 repo or import `model/` locally.
- Self-contained and stable — no dependency on the upstream repo's internal module structure.

**Model configuration:**
```
backbone_scale       = 0.25      # base network at ¼ resolution → fast
refine_mode          = sampling  # fixed-budget patch refinement
refine_sample_pixels = 80,000    # tuned for HD selfie resolution
```

---

### 2. Inference Backend — PyTorch MPS

**Chosen:** `torch.device('mps')` — Apple Metal Performance Shaders

**Why:**
- Native GPU acceleration on Apple Silicon (M1/M2/M3/M4).
- No CUDA/ROCm setup required.
- Achieves ~5 fps on M2 for HD video, which is acceptable for batch-processing offline.
- Automatic CPU fallback if MPS is unavailable.

TorchScript `float32` is used (not `float16`) because MPS has known precision issues with `float16` in some PyTorch versions.

---

### 3. Video I/O — OpenCV (read) + FFmpeg subprocess (write)

**Chosen:** OpenCV for frame reading; FFmpeg subprocess pipe for ProRes encoding.

**Why not use OpenCV's `VideoWriter` for output:**
- OpenCV **cannot write ProRes 4444** with alpha on macOS.
- `cv2.VideoWriter_fourcc('ap4h')` exists but does not support `yuva444p10le` (alpha).
- The only reliable way to produce a true alpha `.mov` is `ffmpeg -c:v prores_ks -profile:v 4444`.

**FFmpeg pipe pattern used:**
```
Python (RGBA uint8 bytes) → stdin pipe → FFmpeg prores_ks → temp .mov
```

**Audio handling:**
1. Audio is demuxed to a temp `.aac` file before the frame loop (fast, codec-copy).
2. After encoding, the ProRes video and audio are remuxed in a second FFmpeg pass.
3. Temp files are cleaned up after completion.

---

### 4. Background Plate Strategy

**Priority order when `--bg-image` is omitted:**

```
1. User-supplied via --bg-image flag
2. Saved image found in assets/ (user is prompted to confirm)
3. Auto-generated via median of first 30 frames
```

**Why median over mean for auto-generation:**
The median pixel value is inherently robust to the subject who is present in most frames — because the subject only occupies a *minority* of pixels at any single location across the 30-frame stack, the median naturally selects the background value.

---

### 5. Output Format — ProRes 4444

**Chosen:** `prores_ks` encoder, `-profile:v 4444`, `-pix_fmt yuva444p10le`

| Property | Value |
|---|---|
| Container | `.mov` (QuickTime) |
| Codec | ProRes 4444 (`prores_ks`) |
| Pixel Format | `yuva444p10le` — 10-bit YUV + alpha |
| Alpha | ✅ True transparency channel |
| Compatibility | CapCut, DaVinci Resolve, FCP, After Effects |

**Why ProRes 4444 over alternatives:**

| Format | Alpha | CapCut | File Size | Quality |
|---|---|---|---|---|
| H.264 `.mp4` | ❌ | ✅ | Small | Lossy |
| ProRes 422 | ❌ | ✅ | Medium | High |
| **ProRes 4444 (chosen)** | **✅** | **✅** | Large | **Lossless** |
| WebM/VP9 | ✅ | ❌ | Small | Lossy |
| PNG sequence | ✅ | ⚠️ (import) | Very large | Lossless |

ProRes 4444 is the only codec that is both **natively supported by CapCut** and carries a **true alpha channel** in a `.mov` container.

---

### 6. Project File Layout

```
video-bg-remover/
├── assets/          ← saved background images (gitignored assets stay local)
├── videos/          ← raw input + processed output videos (gitignored)
├── docs/            ← ADRs and documentation
├── remove_bg.py     ← single-file CLI tool
├── requirements.txt
└── .gitignore
```

**Naming convention:** `{input_stem}_processed.mov` — processed file lives alongside its raw source in `videos/`.

---

## Consequences

### Positive
- High-quality soft alpha matte suitable for professional editing.
- Zero manual model setup — auto-downloaded on first run.
- Clean dependency tree: only `torch`, `torchvision`, `opencv-python`, `numpy`.
- Output is directly import-ready in CapCut, Resolve, FCP.

### Negative / Trade-offs
- BGMv2 **requires a background plate** — auto-generation from median frames works, but a manually captured clean plate gives better results.
- Processing speed is ~5 fps on M2 (not real-time). A future upgrade to Robust Video Matting (RVM) could improve this as RVM doesn't need a background plate and is faster.
- ProRes 4444 files are large (~1–3 GB for a 1-minute HD clip).
- `torch` + `torchvision` add ~500MB to the venv.

---

## Alternatives Considered and Rejected

| Alternative | Reason Rejected |
|---|---|
| MediaPipe Selfie Segmentation | Binary mask only, no true alpha, too many deps |
| Robust Video Matting (RVM) | Excellent model — deferred for future upgrade |
| rembg (U²-Net) | Slow, no temporal consistency for video |
| ONNX Runtime backend | Slower than PyTorch/TorchScript on MPS |
| OpenCV VideoWriter for output | Cannot write ProRes 4444 with alpha |
| PNG frame sequence output | Large, awkward to reimport with audio |

---

## Future Considerations

- **Upgrade to RVM** — does not need a background plate, better temporal consistency.
- **Batch processing** — accept a folder of videos and process in sequence.
- **Watchfolder mode** — auto-process new `.mov` files dropped into `videos/`.
- **Hardware-accelerated decode** — use FFmpeg's `videotoolbox` decoder for faster frame reading on Apple Silicon.
