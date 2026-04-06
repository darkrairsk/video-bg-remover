import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model_if_needed():
    model_path = Path('selfie_segmenter.tflite')
    if not model_path.exists():
        print("Downloading MediaPipe Segmenter Model...")
        url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        urllib.request.urlretrieve(url, str(model_path))
    return str(model_path)

def get_segmenter(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_confidence_masks=True
    )
    return vision.ImageSegmenter.create_from_options(options)

def main():
    parser = argparse.ArgumentParser(description="Remove background from a video using MediaPipe.")
    
    parser.add_argument("input_video", help="Path to the input video file")
    
    parser.add_argument(
        "--format", 
        choices=["video", "png"], 
        default="video", 
        help="Output format: 'video' for processed video, 'png' for transparent image sequence (default: video)"
    )
    
    parser.add_argument(
        "--bg-color",
        type=str,
        default="#00FF00",
        help="Background color in HEX format (e.g., '#FF0000' or '#FF0000FF'). Default is green."
    )
    
    parser.add_argument(
        "--bg-image",
        type=str,
        help="Path to a custom background image."
    )
    
    args = parser.parse_args()
    
    input_video_path = Path(args.input_video).resolve()
    
    if not input_video_path.exists() or not input_video_path.is_file():
        print(f"Error: The video file '{input_video_path}' does not exist.")
        sys.exit(1)
        
    if args.format == "video":
        output_ext = input_video_path.suffix if input_video_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv'] else '.mp4'
        output_path = input_video_path.parent / f"{input_video_path.stem}_processed{output_ext}"
    else: 
        output_path = input_video_path.parent / f"{input_video_path.stem}_processed"
        
    print(f"Input Video  : {input_video_path}")
    print(f"Output Mode  : {args.format.upper()}")
    print(f"Destination  : {output_path}")
    print("-" * 40)
    
    model_path = download_model_if_needed()
    
    if args.format == "video":
        hex_color = args.bg_color.lstrip('#')
        if len(hex_color) not in (6, 8):
            print("Error: Invalid hex color format. Use e.g. '#FF0000'.")
            sys.exit(1)
            
        hex_color = hex_color[:6]  # Use only the RGB part if an alpha channel was supplied
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        replacement_color = (b, g, r) # OpenCV uses BGR
        
        remove_bg_to_video(str(input_video_path), str(output_path), model_path, replacement_color, args.bg_image)
    else:
        remove_bg_to_png(str(input_video_path), str(output_path), model_path)

def remove_bg_to_video(input_path, output_path, model_path, replacement_color=(0, 255, 0), bg_image_path=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if bg_image_path:
        bg_img = cv2.imread(bg_image_path)
        if bg_img is None:
            print(f"Error: Could not load background image '{bg_image_path}'.")
            return
        base_bg_image = cv2.resize(bg_img, (width, height))
        print("Working on Video (Custom Image)...")
    else:
        base_bg_image = np.zeros((height, width, 3), dtype=np.uint8)
        base_bg_image[:] = replacement_color
        print("Working on Video (Solid Color)...")

    frame_count = 0
    
    with get_segmenter(model_path) as segmenter:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = segmenter.segment(mp_image)
            
            # Extract boolean mask (foreground > 0.1)
            mask = result.confidence_masks[0].numpy_view()[:, :, 0] > 0.1
            
            condition = np.stack((mask,) * 3, axis=-1)
            output_frame = np.where(condition, frame, base_bg_image)
            
            out.write(output_frame)
            
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == total_frames:
                print(f"Processed {frame_count} / {total_frames} frames...", end='\r')

    print(f"\nDone! Video saved to: {output_path}")
    cap.release()
    out.release()

def remove_bg_to_png(input_path, output_folder, model_path):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Working on PNG Image Sequence...")
    
    frame_count = 0
    with get_segmenter(model_path) as segmenter:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = segmenter.segment(mp_image)
            
            mask = result.confidence_masks[0].numpy_view()[:, :, 0] > 0.1
            
            b, g, r = cv2.split(frame)
            alpha_channel = np.where(mask, 255, 0).astype(np.uint8)
            transparent_frame = cv2.merge([b, g, r, alpha_channel])
            
            output_file = Path(output_folder) / f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(output_file), transparent_frame)
            
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == total_frames:
                print(f"Processed {frame_count} / {total_frames} frames...", end='\r')

    print(f"\nDone! Saved {frame_count} PNGs to folder: {output_folder}")
    cap.release()

if __name__ == "__main__":
    main()
