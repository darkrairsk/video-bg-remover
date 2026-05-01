# ADR-002: Upgrade to Robust Video Matting (RVM)

| Field        | Value                               |
|--------------|-------------------------------------|
| **Status**   | Accepted                            |
| **Date**     | 2026-04-08                          |
| **Supersedes** | N/A (complements ADR-001)         |
| **Author**   | Rahul                               |

---

## Context

v1 used **BackgroundMattingV2 (BGMv2)** which has two structural limitations for selfie video use:

1. **Flickering edges** — BGMv2 processes frames independently, so soft edges (hair, wisps) produce inconsistent alpha frame-to-frame.
2. **Mandatory background plate** — BGMv2 needs a clean background image. Auto-extraction from 30 frames is an approximation, not a reliable substitute.

For selfie videos destined for YouTube Shorts (hand-held, moving subject), temporal consistency matters more than maximum per-frame sharpness.

---

## Decision

**Default to Robust Video Matting (RVM)** with BGMv2 retained as an opt-in via `--model-type bgmv2`.

### Why RVM

| Property | BGMv2 | RVM |
|---|---|---|
| Background plate required | Yes (strongly) | No |
| Temporal consistency | Low (per-frame) | High (recurrent GRU) |
| Speed on M2 MPS | ~5 fps | ~8–10 fps |
| Model file size | ~20 MB | ~28 MB |
| Input API | `model(src, bgr)` | `model(src, *rec, dr)` |

RVM's recurrent architecture (4 hidden GRU states threaded between frames) means each frame's segmentation is informed by what came before. This eliminates flicker without any post-processing.

### Model variant chosen

`rvm_mobilenetv3_fp32.torchscript` (~28 MB):
- TorchScript variant — self-contained, no upstream repo dependency.
- `float32` — avoids known `float16` precision issues on MPS (PyTorch < 2.1).
- MobileNetV3 backbone — fastest variant; ResNet-50 available via `--model` flag.

### downsample_ratio

RVM requires a `downsample_ratio` scalar tensor that controls the internal recurrent resolution:

```
ratio = min(1.0, target / min(W, H))
```

Target of 512px on the shorter side gives a good balance of speed and matting quality for HD selfie footage (~0.47 for 1080p).

### Recurrent state management

```python
rec = [None] * 4          # init: None on the first frame
for frame in frames:
    out = model(src, *rec, downsample_ratio)
    fgr, pha = out[0], out[1]
    rec = list(out[2:])   # carry forward for the next frame
```

The hidden state must be carried forward across every frame. Starting it fresh (via `[None]*4`) causes temporal instability for the first few frames (model "wakes up"); this is expected and brief.

---

## BGMv2 Retention Policy

BGMv2 is kept accessible via `--model-type bgmv2` for:
- Controlled studio shots where a clean background plate is available.
- Users who prefer the slightly sharper per-frame matting BGMv2 can produce.

The `--bg-image` flag remains accepted for both model types; RVM can optionally use it (it is ignored in the current pipeline but could be used as a hint in a future version).

---

## Consequences

### Positive
- No background plate required by default — plug-and-play workflow.
- Significantly smoother alpha at hair/edges across video frames.
- Faster inference on M2 MPS.

### Negative
- If the very first frame has a complex scene, RVM may take 2-3 frames to "warm up" before the alpha stabilises. This is typically imperceptible in videos > 1 second.
- Model file slightly larger (28 MB vs 20 MB).

---

## Future

- Consider `rvm_resnet50_fp32.torchscript` for maximum quality when processing time is not a constraint.
- Evaluate fusing an optional background hint into the RVM pipeline for hybrid quality.
