import argparse
import sys
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Remove background from a video using MediaPipe.")
    
    # Requirement 2: Terminal command takes path to video
    parser.add_argument("input_video", help="Path to the input video file")
    
    # Requirement 1 & 3: Configurable options, default is mp4
    parser.add_argument(
        "--format", 
        choices=["mp4", "png"], 
        default="mp4", 
        help="Output format: 'mp4' for green screen video, 'png' for transparent image sequence (default: mp4)"
    )
    
    parser.add_argument(
        "--bg-color",
        type=str,
        default="#00FF00",
        help="Background color in HEX format (e.g., '#FF0000'). Default is green."
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
        
    # Requirement 4: Output in same folder, with _processed added
    if args.format == "mp4":
        output_path = input_video_path.parent / f"{input_video_path.stem}_processed.mp4"
    else: # png
        output_path = input_video_path.parent / f"{input_video_path.stem}_processed"
        
    print(f"Input Video  : {input_video_path}")
    print(f"Output Mode  : {args.format.upper()}")
    print(f"Destination  : {output_path}")
    print("-" * 40)
    
    if args.format == "mp4":
        hex_color = args.bg_color.lstrip('#')
        if len(hex_color) != 6:
            print("Error: Invalid hex color format. Use e.g. '#FF0000'.")
            sys.exit(1)
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        replacement_color = (b, g, r) # OpenCV uses BGR
        
        remove_bg_to_mp4(str(input_video_path), str(output_path), replacement_color, args.bg_image)
    else:
        remove_bg_to_png(str(input_video_path), str(output_path))

def remove_bg_to_mp4(input_path, output_path, replacement_color=(0, 255, 0), bg_image_path=None):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
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
        print("Working on MP4 (Custom Image)...")
    else:
        base_bg_image = np.zeros((height, width, 3), dtype=np.uint8)
        base_bg_image[:] = replacement_color
        print("Working on MP4 (Solid Color)...")

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)
        
        # You can adjust this threshold (0.1) if you find it's cutting out too much or too little of you
        mask = results.segmentation_mask > 0.1
        
        condition = np.stack((mask,) * 3, axis=-1)
        output_frame = np.where(condition, frame, base_bg_image)
        
        out.write(output_frame)
        
        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count} / {total_frames} frames...", end='\r')

    print(f"\nDone! Video saved to: {output_path}")
    cap.release()
    out.release()

def remove_bg_to_png(input_path, output_folder):
    from pathlib import Path
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Working on PNG Image Sequence...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)
        
        mask = results.segmentation_mask > 0.1
        b, g, r = cv2.split(frame)
        alpha_channel = np.where(mask, 255, 0).astype(np.uint8)
        transparent_frame = cv2.merge([b, g, r, alpha_channel])
        
        # Save as a formatted string: frame_00000.png, frame_00001.png
        output_file = Path(output_folder) / f"frame_{frame_count:05d}.png"
        cv2.imwrite(str(output_file), transparent_frame)
        
        frame_count += 1
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count} / {total_frames} frames...", end='\r')

    print(f"\nDone! Saved {frame_count} PNGs to folder: {output_folder}")
    cap.release()

if __name__ == "__main__":
    main()
