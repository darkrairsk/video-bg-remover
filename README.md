# Video Background Remover

AI-powered background removal for selfie videos using **Robust Video Matting (RVM)** and **BackgroundMattingV2**. Outputs a **ProRes 4444 `.mov`** with a true alpha channel — ready to drop into CapCut, DaVinci Resolve, or Final Cut Pro for YouTube Shorts editing.

Optimised for **Apple Silicon (Mac M2/M3/M4)** via the PyTorch MPS backend.

---

## Features

- 🧠 **Robust Video Matting (RVM)** & **BackgroundMattingV2** — state-of-the-art soft alpha matting
- 🍎 **Apple MPS acceleration** — GPU inference on Apple Silicon, auto-falls back to CPU
- 🎬 **ProRes 4444 with Alpha** — true transparency, not green-screen
- 🔊 **Audio preserved** — original audio remuxed into the final file
- 📐 **Auto-downscale & Internal Resolution** — scale for fast processing (`--internal-res`)
- 🧹 **Guided Filter Smoothing** — edge smoothing using OpenCV guided filter (`--guided-filter`)
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

> **Model auto-download:** On the very first run the script automatically downloads the requested model (`rvm`, `rvm_resnet50`, or `bgmv2`) from GitHub Releases into the project root. No manual setup needed.

---

## CLI Reference

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | ✅ Yes | — | Path to the input `.mov` / `.mp4` video |
| `--model-type` | ❌ No | `rvm` | `rvm` (MobileNetV3), `rvm_resnet50`, or `bgmv2` |
| `--bg-image` | ❌ No | Auto-detected | Path to a clean background image (required for bgmv2, optional for rvm) |
| `--output` | ❌ No | `videos/<name>_processed.mov` | Custom path for the output `.mov` |
| `--internal-res` | ❌ No | `512` | Target short-side resolution for inference |
| `--guided-filter` | ❌ No | `False` | Apply OpenCV Guided Filter for smoother edges |

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
  --model-type rvm_resnet50 \
  --internal-res 720 \
  --guided-filter
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

### 6 — Pick an AI Model

```bash
python remove_bg.py --input videos/selfie.mov --model-type rvm_resnet50
```

Available model types (auto-downloaded on first use):

| `--model-type` | Backbone | Speed | Quality |
|---|---|---|---|
| `rvm` (default) | MobileNetV3 | ⚡ Very Fast | Good |
| `rvm_resnet50` | ResNet-50 | 🐢 Slower | Best |
| `bgmv2` | MobileNetV2 | ⚡ Fast | Good (requires bg plate) |

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
│   ├── ADR-001-bgmv2-prores-pipeline.md  ← Architecture Decision Record
│   └── ADR-002-rvm-upgrade.md            ← RVM upgrade decisions
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
| `opencv-contrib-python` | Frame reading, resizing, guided filter smoothing |
| `imageio-ffmpeg` | Robust FFmpeg binary fetching |
| `numpy` | Tensor ↔ numpy conversion, RGBA composition |

---

## Architecture Decisions

See [`docs/ADR-001-bgmv2-prores-pipeline.md`](docs/ADR-001-bgmv2-prores-pipeline.md) and [`docs/ADR-002-rvm-upgrade.md`](docs/ADR-002-rvm-upgrade.md) for rationale on every major technical choice (model selection, MPS backend, FFmpeg pipe, ProRes 4444, background plate strategy).

---

## License

MIT
