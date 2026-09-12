import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

# -------------------------------------------------------------------
# 1. Configuration & Client Initialization
# -------------------------------------------------------------------
GEMINI_API_KEY = ""
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable missing.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Use Google's dedicated Virtual Try-On recontextualization endpoint
TRYON_MODEL = "virtual-try-on"  


# -------------------------------------------------------------------
# 2. Camera Recording Helper
# -------------------------------------------------------------------
def record_webcam_video(output_video_path: str = "recorded_input.mp4", fps: int = 30) -> str:
    """
    Opens the default camera, previews the feed, and records video between
    presses of the SPACE key. Returns the path of the saved video.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access the camera.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    is_recording = False
    print("\n[Webcam Ready] Press 'SPACE' to START recording. Press 'SPACE' again to STOP.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Mirror the live feed visually for easier framing
        display_frame = frame.copy()

        if is_recording:
            # Write original frame to file
            out.write(frame)
            # Add visual indicator to live preview
            cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC - Press SPACE to Stop", (55, 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press SPACE to Record", (30, 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Webcam - Virtual Try-On Recorder", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE key to toggle recording
            if not is_recording:
                is_recording = True
                print(" -> Recording started...")
            else:
                is_recording = False
                print(" -> Recording stopped.")
                break
        elif key == ord('q'):  # Press 'q' to abort
            print(" -> Recording cancelled.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
        raise RuntimeError("No video was recorded.")

    return output_video_path


# -------------------------------------------------------------------
# 3. Virtual Try-On Core Request
# -------------------------------------------------------------------
def process_tryon_frame(person_frame_bgr: np.ndarray, garment_img: Image.Image) -> np.ndarray:
    """
    Sends a person frame + garment image to Google Cloud Virtual Try-On model
    and returns the reconstructed BGR image frame.
    """
    # Convert OpenCV BGR array to PIL Image
    frame_rgb = cv2.cvtColor(person_frame_bgr, cv2.COLOR_BGR2RGB)
    person_img = Image.fromarray(frame_rgb)

    try:
        # Request recontextualization from Google Virtual Try-On model
        response = client.models.recontext_image(
            model=TRYON_MODEL,
            source=types.RecontextImageSource(
                person_image=person_img,
                product_images=[
                    types.ProductImage(product_image=garment_img)
                ],
            ),
            config=types.RecontextImageConfig(
                output_mime_type="image/jpeg",
                number_of_images=1,
                safety_filter_level="BLOCK_LOW_AND_ABOVE",
            ),
        )

        # Retrieve generated try-on image
        if response.generated_images:
            result_pil = response.generated_images[0].image
            result_rgb = np.array(result_pil)
            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"[API Warning] Frame processing failed: {e}")

    # Return original frame as fallback if API call fails
    return person_frame_bgr


# -------------------------------------------------------------------
# 4. Video Processing Pipeline
# -------------------------------------------------------------------
def process_video_tryon(
    video_path: str, 
    garment_path: str, 
    output_path: str, 
    frame_skip: int = 5
):
    """
    Reads a video file frame-by-frame, runs Try-On on every N-th frame,
    and saves the output to a new MP4 video.
    """
    if not os.path.exists(video_path) or not os.path.exists(garment_path):
        raise FileNotFoundError("Input video or garment image does not exist.")

    garment_img = Image.open(garment_path).convert("RGB")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Video specs
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n[Start] Processing recorded video ({total_frames} total frames, FPS: {fps})...")
    
    frame_count = 0
    last_processed_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame via Google Try-On API every N frames to save API time
        if frame_count % frame_skip == 0:
            print(f" -> Rendering Virtual Try-On for Frame {frame_count}/{total_frames}...")
            start_t = time.time()
            
            processed = process_tryon_frame(frame, garment_img)
            last_processed_frame = cv2.resize(processed, (width, height))
            
            print(f"    Done in {time.time() - start_t:.2f}s")

        # Write frame to video stream
        out.write(last_processed_frame if last_processed_frame is not None else frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[Success] Video Virtual Try-On complete! Output saved to: {output_path}")


# -------------------------------------------------------------------
# 5. Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    RECORDED_VIDEO = "webcam_input.mp4"
    GARMENT_IMAGE = "images.jpeg"
    OUTPUT_VIDEO = "tryon_result.mp4"

    # Step 1: Record video live from webcam
    # recorded_file = record_webcam_video(output_video_path=RECORDED_VIDEO)

    # Step 2: Process recorded video through Virtual Try-On model
    process_video_tryon(
        video_path=RECORDED_VIDEO,
        garment_path=GARMENT_IMAGE,
        output_path=OUTPUT_VIDEO,
        frame_skip=5
    )