import os
import gradio as gr
import cv2
import numpy as np
from ghost import ghost_batch_process_image, get_model
from fsw_util import get_face_analyser, upsample, export_as_gif
from torchvision import transforms
from PIL import Image
from util import batch_process_image

def tensor_to_image(tensor):
    image = tensor.clone().detach()
    image = image.numpy().transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

def normalize_images(image_batch):
    """
    Normalize a batch of images as though they had been read via cv2.imread.

    Args:
        image_batch (list of np.ndarray): A list of images to normalize.

    Returns:
        list of np.ndarray: A list of normalized images.
    """
    # Define the normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    normalized_images = []
    
    for image in image_batch:
        # Convert the image from BGR (cv2.imread format) to RGB (PIL format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the image to a PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply the transformation (including normalization)
        normalized_tensor = transform(pil_image)
        
        # Convert the tensor back to an image format
        normalized_image_np = tensor_to_image(normalized_tensor)
        
        # Convert to BGR format if needed (optional)
        normalized_image_bgr = cv2.cvtColor((normalized_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        normalized_images.append(normalized_image_bgr)
    
    return normalized_images

def process_video_with_image(image_path, video_path, output_path="/tmp/", batch_size=8):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    source_image = cv2.imread(image_path)
    if source_image is None:
        print(f"Error: Cannot read image file {image_path}")
        return
    
    source_image = normalize_images([source_image])[0]

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
                frame_buffer = normalize_images(frame_buffer)
                processed_frames = ghost_batch_process_image(np.array(frame_buffer), source_image)
                for processed_frame in processed_frames:
                    out.write(processed_frame[:,:,::-1])
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:
            # Process the batch of frames
            frame_buffer = normalize_images(frame_buffer)
            processed_frames = ghost_batch_process_image(np.array(frame_buffer), source_image)
            for processed_frame in processed_frames:
                out.write(processed_frame[:,:,::-1])
            frame_buffer = []

    video.release()
    out.release()
    print(f"Output saved to {output_file}")
    return output_file

def process_video_with_image_128(image_path, video_path, output_path="/tmp/", batch_size=16):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    source_image = cv2.imread(image_path)
    if source_image is None:
        print(f"Error: Cannot read image file {image_path}")
        return
    
    source_image = normalize_images([source_image])[0]

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
                frame_buffer = normalize_images(frame_buffer)
                processed_frames = batch_process_image(np.array(frame_buffer), source_image, False)
                for processed_frame in processed_frames:
                    out.write(processed_frame[:,:,::-1])
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:
            # Process the batch of frames
            frame_buffer = normalize_images(frame_buffer)
            processed_frames = batch_process_image(np.array(frame_buffer), source_image, False)
            for processed_frame in processed_frames:
                out.write(processed_frame[:,:,::-1])
            frame_buffer = []

    video.release()
    out.release()
    print(f"Output saved to {output_file}")
    return output_file

def handle_image_upsample(file):
    file_path = file.name
    file_dir = os.path.dirname(file_path)
    save_path = os.path.join(file_dir, f"upsampled_{os.path.basename(file_path)}")

    if not os.path.exists(save_path):
        image = cv2.imread(file_path)
        final_face = upsample(image)
        final_face = Image.fromarray(final_face[:,:,::-1])
        final_face.save(save_path)
    return save_path, gr.update(visible=True), save_path

def handle_video_upsample(file):
    file_path = file.name
    file_dir = os.path.dirname(file_path)
    save_path = os.path.join(file_dir, f"upsampled_{os.path.basename(file_path)}")

    if not os.path.exists(save_path):
        video = cv2.VideoCapture(file_path)

        if not video.isOpened():
            raise ValueError(f"Error: Cannot open video file {file_path}")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width * 2, frame_height * 2))

        while True:
            ret, frame = video.read()
            if not ret:
                break

            upsampled_frame = upsample(frame)
            out.write(upsampled_frame)

        video.release()
        out.release()

    print(f"Upsample saved to {save_path}")
    return save_path, gr.update(visible=True), save_path

def list_all_files(directory="/app/ghost_models"):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().startswith('g_'):
                file_paths.append(os.path.join(root, file))
    return sorted(file_paths)

def process_selected_file(path):
    if os.path.basename(path).lower().startswith("g_"):
        model=get_model(model_file=path, force_load=True)
        return f"Loaded {path}"
    else:
        return "Please select a model file (G_latest.pth or G_#_####.pth)."

# Define the Gradio interface
with gr.Blocks() as iface:
    # Tab 1: Processing
    with gr.Tab("Processing"):
        image_input = gr.Image(type="filepath", label="Upload Image")
        video_input = gr.File(label="Upload Video or GIF", file_types=["video", "image"])
        output_video = gr.Video(label="Processed Video")
        process_button = gr.Button("Process")
        process_button.click(
            process_video_with_image,
            inputs=[image_input, video_input],
            outputs=output_video
        )

    # Tab 2: Image Upsampling
    with gr.Tab("Image Upsampling"):
        image_upsample_input = gr.File(label="Upload Image", file_types=["image"])
        image_upsample_result = gr.Image(visible=False)
        send_to_processing_image = gr.Button("Send to Processing Tab", visible=False)
        image_path_state = gr.State()
        image_upsample_button = gr.Button("Upsample Image")

        def display_image_output(save_path):
            return gr.update(value=save_path, visible=True), gr.update(visible=True), save_path

        image_upsample_button.click(
            handle_image_upsample,
            inputs=image_upsample_input,
            outputs=[image_upsample_result, send_to_processing_image, image_path_state]
        )

        send_to_processing_image.click(
            lambda x: gr.update(value=x),
            inputs=image_path_state,
            outputs=image_input,
        )

    # Tab 3: Video Upsampling
    with gr.Tab("Video Upsampling"):
        video_upsample_input = gr.File(label="Upload Video", file_types=["video", "image"])
        video_upsample_result = gr.Video(visible=False)
        send_to_processing_video = gr.Button("Send to Processing Tab", visible=False)
        video_path_state = gr.State()
        video_upsample_button = gr.Button("Upsample Video")

        def display_video_output(save_path):
            return gr.update(value=save_path, visible=True), gr.update(visible=True), save_path

        video_upsample_button.click(
            handle_video_upsample,
            inputs=video_upsample_input,
            outputs=[video_upsample_result, send_to_processing_video, video_path_state]
        )

        send_to_processing_video.click(
            lambda x: gr.update(value=x),
            inputs=video_path_state,
            outputs=video_input,
        )
    with gr.Tab("Select Model"):
        # Generate the list of all files
        all_files = list_all_files()
        
        # Dropdown to select a file
        file_selector = gr.Dropdown(label="Select a Model", choices=all_files)
        
        # Textbox to display the selected file
        output_text = gr.Textbox(label="Output")
        
        # Logic for processing the selected file
        file_selector.change(
            fn=process_selected_file, 
            inputs=file_selector, 
            outputs=output_text
        )
    # Tab 1: Processing
    with gr.Tab("Processing (128x128)"):
        image_input_128 = gr.Image(type="filepath", label="Upload Image")
        video_input_128 = gr.File(label="Upload Video or GIF", file_types=["video", "image"])
        output_video_128 = gr.Video(label="Processed Video")
        process_button_128 = gr.Button("Process")
        process_button_128.click(
            process_video_with_image_128,
            inputs=[image_input_128, video_input_128],
            outputs=output_video_128
        )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=5000, share=False)
