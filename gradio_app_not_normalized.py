import os
import gradio as gr
import cv2
import numpy as np
from ghost import ghost_batch_process_image
from fsw_util import get_face_analyser

def process_video_with_image(image_path, video_path, output_path="/tmp/", batch_size=8):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    source_image = cv2.imread(image_path)
    if source_image is None:
        print(f"Error: Cannot read image file {image_path}")
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video Resolution {frame_width}x{frame_height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    output_file = os.path.join(output_path, f"output_{os.path.basename(image_path)}.mp4")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    frame_buffer = []
    
    while True:
        ret, frame = video.read()

        if not ret:
            if frame_buffer:
                # Process any remaining frames in the buffer
                processed_frames = ghost_batch_process_image(np.array(frame_buffer), source_image)
                for processed_frame in processed_frames:
                    out.write(processed_frame[:,:,::-1])
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:
            # Process the batch of frames
            processed_frames = ghost_batch_process_image(np.array(frame_buffer), source_image)
            for processed_frame in processed_frames:
                out.write(processed_frame[:,:,::-1])
            frame_buffer = []

    video.release()
    out.release()
    print(f"Output saved to {output_file}")
    return output_file


# Define the Gradio interface
iface = gr.Interface(
    fn=process_video_with_image,
    inputs=[gr.Image(type="filepath"), 
            gr.File(label="Upload Video or GIF", file_types=["video", "image"])
    ],
    outputs="video",
    title="Image and Video Processing"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=5000, share=False)
