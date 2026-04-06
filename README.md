# Video Background Remover

A simple Python tool to remove the background of videos using MediaPipe and replace it with a solid color or a custom image. It isolates the subject and replaces the background seamlessly.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/video-bg-remover.git
cd video-bg-remover
```

2. Set up the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run `remove_bg.py` by providing the path to your input video.

### Basic Usage:
```bash
python remove_bg.py path/to/your/video.mov
```
This will generate a video with a solid green background (Green Screen) by default. The script supports common video formats like `.mp4`, `.mov`, and `.avi`. The output will be saved in the same directory as the input video with `_processed` appended to the filename and the same extension as the original video.

### Using a Custom Background Color:
You can specify a custom background color using a HEX code (both 6-character like `#FF0000` and 8-character with alpha like `#0adb3eff` are supported).
```bash
python remove_bg.py user_video.mov --bg-color "#0adb3eff"
```

### Using a Custom Background Image:
You can replace the background with an image. The script will resize the image to fit your video's resolution.
```bash
python remove_bg.py user_video.mov --bg-image beautiful_scenery.jpg
```

### Transparent Image Sequence (PNG):
If you need a perfectly transparent background, you can output a sequence of PNG frames.
```bash
python remove_bg.py user_video.mov --format png
```

## How It Works
We use the **MediaPipe Tasks API (ImageSegmenter)** to intelligently separate the foreground subject from the background dynamically. The script will automatically download the required `selfie_segmenter.tflite` model directly from Google upon its very first run, so no manual setup of the ML model is required.
