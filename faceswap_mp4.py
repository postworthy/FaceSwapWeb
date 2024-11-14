import os
import cv2
import sys
import time
import numpy as np
from ghost import ghost_batch_process_image
from fsw_util import get_face_analyser

def load_images_from_directory(image_dir):
    """Load all images from the specified directory."""
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, file_name)
            #image = cv2.imread(image_path)
            #if image is not None:
            images.append(image_path)
    return images

def preprocess_video(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video Resolution {frame_width}x{frame_height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    output_file = os.path.join(output_path, f"pre_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()

        if not ret:
            break

        faces = sorted(get_face_analyser().get(frame), key=lambda x: x.bbox[0])

        if faces:
            out.write(frame)
            
    video.release()
    out.release()
    print(f"Output saved to {output_file}")
    return output_file

def process_video_with_image(video_path, image_path, output_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    source_image = cv2.imread(image_path)
    if source_image is None:
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video Resolution {frame_width}x{frame_height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    output_file = os.path.join(output_path, f"output_{os.path.basename(image_path)}.mp4")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        faces = sorted(get_face_analyser().get(frame), key=lambda x: x.bbox[0])
        if faces:
            processed_frame = ghost_batch_process_image(np.array([frame]), source_image)[0][:,:,::-1]        
            out.write(processed_frame)
            
    video.release()
    out.release()
    print(f"Output saved to {output_file}")


def main(video_path='input.mp4', image_dir='/app/celeb', output_dir='/app/output'):
    images = load_images_from_directory(image_dir)
    if not images:
        print(f"Error: No images found in directory {image_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocessed_video_path = preprocess_video(video_path, output_dir)

    for image_path in images:
        print(f"Processing with image: {image_path}")
        process_video_with_image(preprocessed_video_path, image_path, output_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print("Usage: python faceswap_mp4.py [video_path] [image_directory] [output_directory]")
        print("Defaults:")
        print("  video_path: input.mp4")
        print("  image_directory: /app/celeb")
        print("  output_directory: /app/output")
        sys.exit(1)

    video_path = sys.argv[1] if len(sys.argv) > 1 else '/app/videos/input.mp4'
    image_dir = sys.argv[2] if len(sys.argv) > 2 else '/app/celeb/'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '/app/output/'
    main(video_path, image_dir, output_dir)
