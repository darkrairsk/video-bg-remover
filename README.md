# Video Background Remover

AI-powered background removal for selfie videos using **BackgroundMattingV2**. Outputs a **ProRes 4444 `.mov`** with a true alpha channel — ready to drop into CapCut, DaVinci Resolve, or Final Cut Pro for YouTube Shorts editing.

Optimised for **Apple Silicon (Mac M2/M3/M4)** via the PyTorch MPS backend.

---

## Features

- 🧠 **BackgroundMattingV2** — state-of-the-art soft alpha matting (CVPR 2021)
- 🍎 **Apple MPS acceleration** — GPU inference on Apple Silicon, auto-falls back to CPU
- 🎬 **ProRes 4444 with Alpha** — true transparency, not green-screen
- 🔊 **Audio preserved** — original audio remuxed into the final file
- 📐 **Auto-downscale** — videos larger than 1080p are scaled down for fast processing
- 🖼 **Smart background detection** — checks `assets/` for a saved plate before extracting from frames
- 📁 **Organised output** — raw + processed videos live together in `videos/`

---

## Prerequisites

### 1 — System dependency (FFmpeg)

```bash
brew install ffmpeg
```

### 2 — Python environment

```bash
# Clone the repo
git clone https://github.com/your-username/video-bg-remover.git
cd video-bg-remover

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

> **Model auto-download:** On the very first run the script automatically downloads the BGMv2 TorchScript model (~15 MB) from GitHub Releases into the project root. No manual setup needed.

---

## CLI Reference

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | ✅ Yes | — | Path to the input `.mov` / `.mp4` video |
| `--bg-image` | ❌ No | Auto-detected | Path to a clean background image (see Background Plate logic below) |
| `--output` | ❌ No | `videos/<name>_processed.mov` | Custom path for the output `.mov` |
| `--model` | ❌ No | `./torchscript_mobilenetv2_fp32.pth` | Path to a custom TorchScript model file |

---

## Usage Examples

### 1 — Simplest run (auto everything)

```bash
python remove_bg.py --input videos/selfie.mov
```

- Output: `videos/selfie_processed.mov`
- Background: checks `assets/` for a saved image, then prompts; if none found, auto-generates from the first 30 frames of the video.

---

### 2 — Supply a clean background image

Best results. Take a photo of your background *without you in it* and pass it directly:

```bash
python remove_bg.py --input videos/selfie.mov --bg-image assets/bg_image.jpg
```

- Output: `videos/selfie_processed.mov`
- Background: uses the supplied image, resized to match the video.

---

### 3 — Custom output path

```bash
python remove_bg.py --input videos/selfie.mov --output /Volumes/SSD/exports/clean.mov
```

- Output: saved to the specified path.
- Background: auto-detected from `assets/` or extracted from frames.

---

### 4 — Full explicit run (all arguments)

```bash
python remove_bg.py \
  --input   videos/selfie.mov \
  --bg-image assets/bg_image.jpg \
  --output  videos/selfie_processed.mov \
  --model   torchscript_mobilenetv2_fp32.pth
```

---

### 5 — Input from outside the project (video gets copied in)

You can point `--input` at any path on your machine. The script copies the raw video into `videos/` automatically:

```bash
python remove_bg.py --input ~/Downloads/clip.mov --bg-image assets/bg_image.jpg
```

- Raw video copied to: `videos/clip.mov`
- Output:             `videos/clip_processed.mov`

---

### 6 — Use a custom / alternative model

```bash
python remove_bg.py \
  --input videos/selfie.mov \
  --model /path/to/torchscript_resnet50_fp32.pth
```

Available BGMv2 model variants from the [GitHub Releases](https://github.com/PeterL1n/BackgroundMattingV2/releases/tag/v1.0.0):

| File | Backbone | Speed | Quality |
|---|---|---|---|
| `torchscript_mobilenetv2_fp32.pth` | MobileNetV2 | ⚡ Fast | Good |
| `torchscript_resnet50_fp32.pth` | ResNet-50 | 🐢 Slower | Better |

---

## Background Plate Logic

When `--bg-image` is **not** supplied, the tool follows this priority:

```
1. Check assets/ for any saved image (jpg, jpeg, png, bmp, webp)
   └─ Found → prompt: "Use this background? [Y/n]"
        ├─ Y → use it
        └─ n → fall through to step 2

2. Auto-generate from video
   └─ Read first 30 frames → compute per-pixel median → use as plate
```

**Tip:** Save your background photo to `assets/bg_image.jpg` once — the tool will find and offer it on every future run.

---

## Output Format

| Property | Value |
|---|---|
| Container | `.mov` (QuickTime) |
| Video Codec | ProRes 4444 (`prores_ks`) |
| Pixel Format | `yuva444p10le` (10-bit + alpha) |
| Audio Codec | AAC (remuxed from original) |
| Transparency | ✅ True alpha channel |

**Compatible with:** CapCut · DaVinci Resolve · Final Cut Pro · After Effects · Premiere Pro

---

## Project Structure

```
video-bg-remover/
├── assets/                              ← Save your background image here
│   └── bg_image.jpg
├── videos/                              ← Raw + processed videos (gitignored)
│   ├── another_sample.mov
│   └── another_sample_processed.mov
├── docs/
│   └── ADR-001-bgmv2-prores-pipeline.md  ← Architecture Decision Record
├── remove_bg.py                         ← Main CLI script
├── requirements.txt                     ← Python dependencies
├── torchscript_mobilenetv2_fp32.pth     ← Auto-downloaded model (gitignored)
└── .gitignore
```

---

## How It Works

1. **Background Plate** — provided, saved in `assets/`, or auto-generated via median filtering.
2. **Audio Extraction** — FFmpeg demuxes the original audio to a temp `.aac`.
3. **AI Matting** — each frame + background plate pass through BGMv2 on the MPS GPU → returns `pha` (alpha matte) and `fgr` (foreground RGB).
4. **RGBA Composition** — `R,G,B = fgr × alpha` (pre-multiplied), `A = alpha`. 4-channel frame piped to FFmpeg.
5. **ProRes Encoding** — `ffmpeg -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le`.
6. **Audio Remux** — ProRes video + temp audio merged; temp files deleted.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | PyTorch — model inference + MPS backend |
| `torchvision` | Vision utilities (required by BGMv2) |
| `opencv-python` | Frame reading from video |
| `numpy` | Tensor ↔ numpy conversion, RGBA composition |
| `ffmpeg` *(system)* | ProRes encoding, audio demux/remux |

---

## Architecture Decisions

See [`docs/ADR-001-bgmv2-prores-pipeline.md`](docs/ADR-001-bgmv2-prores-pipeline.md) for rationale on every major technical choice (model selection, MPS backend, FFmpeg pipe, ProRes 4444, background plate strategy).

---

## License

MIT
