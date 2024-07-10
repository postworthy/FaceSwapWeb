import os
import cv2
os.environ['OMP_NUM_THREADS'] = '1'
import threading
from PIL import Image
import torch
import insightface
import onnxruntime
from flask_socketio import SocketIO, send
from collections import deque
import py7zr
import sys
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
import clip as clip
#from interrogate import InterrogateModels
from fsw_util import push_action, export_as_gif, export_as_jpg, get_face_single, get_face_swapper, process_frames, get_fps, get_face_analyser
from insightface.utils import face_align

#interrogator = InterrogateModels("interrogate")

THREAD_LOCK = threading.Lock()
FACE_SWAPPER = None
FACE_ANALYSER = None

zip_path = "/tmp/static.7z"
zip_pass = ""
face_id = '1.jpg'
source_face = {}
    
def search_files(file_handler, directory = './static/videos'):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp4', '.gif')):
                file_handler(os.path.join(root, file))

def expand_faces(faces, expansion=5, img_width=None, img_height=None):
    """
    Expands the bounding boxes of detected faces.

    Args:
    - faces: List of Face objects detected by the FaceAnalysis.
    - expansion: The number of pixels to expand the bounding box by on each side.
    - img_width, img_height: Dimensions of the image to ensure expanded bbox doesn't exceed image boundaries.

    Returns:
    - List of expanded bounding boxes.
    """
    expanded_bboxes = []
    for face in faces:
        bbox = face.bbox
        # Calculate expanded bbox coordinates
        x_expanded = max(0, bbox[0] - expansion)
        y_expanded = max(0, bbox[1] - expansion)
        w_expanded = bbox[2] - bbox[0] + 2 * expansion
        h_expanded = bbox[3] - bbox[1] + 2 * expansion

        # Ensure the expanded bounding box does not exceed image boundaries
        if img_width is not None and img_height is not None:
            x_expanded = min(max(0, x_expanded), img_width - 1)
            y_expanded = min(max(0, y_expanded), img_height - 1)
            w_expanded = min(w_expanded, img_width - x_expanded)
            h_expanded = min(h_expanded, img_height - y_expanded)

        expanded_bboxes.append([x_expanded, y_expanded, w_expanded, h_expanded])
    return expanded_bboxes

def extract_regions_from_image(image, expanded_bboxes):
    """
    Extracts regions from the image based on the expanded bounding boxes, ensuring coordinates are integers.

    Args:
    - image: The original image as a numpy array.
    - expanded_bboxes: A list of expanded bounding boxes [x, y, w, h].

    Returns:
    - A list of the extracted image regions as numpy arrays.
    """
    regions = []
    for bbox in expanded_bboxes:
        # Convert bbox coordinates to integers
        x, y, w, h = map(int, bbox)
        # Ensure the bounding box is within image dimensions
        x_end = min(x + w, image.shape[1])
        y_end = min(y + h, image.shape[0])
        # Crop and save the region
        region = image[y:y_end, x:x_end]
        regions.append(region)
    return regions




def extract_faces(file_path):
    global face_id
    global source_face
    print("Processing file:", file_path)
    
    if file_path.endswith('.mp4') or file_path.endswith('.gif'):
        print(file_path)
        video = cv2.VideoCapture(file_path)
        vid_id = os.path.splitext(os.path.basename(file_path))[0]
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not frame_count % fps == 0:
                frame_count += 1
                continue

            if not ret:
                break
            faces = get_face_analyser().get(frame)

            if face_id == None or source_face == None:
                for i, face in enumerate(faces):
                    save_file = f"./static/extracted/swapped_face_f0_v{vid_id}_i{frame_count}_{i}.jpg"
                    if face:
                        warped_img, _ = face_align.norm_crop2(frame, face.kps, 256)
                        Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_RGBA2BGR)).save(save_file)
            else:
                for i, face in enumerate(faces):
                    face_width = face.bbox[2] - face.bbox[0]  # Calculate the width of the face bounding box
                    if face_width < 25:
                        continue
                    
                    swapped_face = get_face_swapper().get(frame, face, source_face, paste_back=True)
                    
                    _, jpeg_frame = cv2.imencode('.jpg', swapped_face)
                    decoded_frame = cv2.imdecode(np.frombuffer(jpeg_frame, np.uint8), cv2.IMREAD_COLOR)
                    rgb_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)

                    face_img = face.bbox.astype(int)
                    cropped_face = rgb_frame[face_img[1]:face_img[3], face_img[0]:face_img[2], :]
                    
                    try:
                        cropped_pil = Image.fromarray(cropped_face)
                    except Exception as e:
                        print(e)
                        continue

                    save_file = f"./static/extracted/swapped_face_f{face_id}_v{vid_id}_i{frame_count}_{i}.jpg"
                    #swapped_face.save(save_file)  
                    cropped_pil.save(save_file)  
                    print(save_file)
            
            frame_count += 1


        video.release()

def extract(password):
    try:
        with py7zr.SevenZipFile(zip_path, mode='r', password=password) as archive:
            zip_pass = password
            image_dir = './static/images'
            video_dir = './static/videos'
            images = sorted(os.listdir(image_dir), key=lambda filename: int(filename.split('.')[0]))
            videos = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
            if len(images) == 0 or len(videos) == 0:
                archive.extractall(path="./static/")
    except Exception as e: 
        print(e)
        return False
    else:
        return True

#def create_captions(input_directory = './static/extracted/'):
#    # Process each file in the directory
#    for file_name in os.listdir(input_directory):
#        if file_name.endswith((".jpg", ".jpeg", ".png")):
#            file_path = os.path.join(input_directory, file_name)
#
#            print("Interrogating", file_path)
#            caption = interrogator.interrogate(Image.open(file_path))
#            print("Interrogating Completed", file_path)
#            torch.cuda.empty_cache()
#
#            # Save the caption to a file
#            output_file_name = os.path.splitext(file_name)[0] + ".caption"
#            output_file_path = os.path.join(input_directory, output_file_name)
#            with open(output_file_path, "w") as output_file:
#                output_file.write(caption)
#                print("Saved", output_file_name)


#docker run -it --gpus all -v ./static/extracted/:/app/static/extracted -v ./static/static.bin:/tmp/static.7z oneshot-faceswap-web:latest python3 face-extraction.py PASSWD 1.jpg
#docker run -it --gpus all -v ./static/extracted/:/app/static/extracted -v ./static/static.bin:/tmp/static.7z oneshot-faceswap-web:latest python3 face-extraction.py PASSWD
if __name__ == '__main__':
    if len(sys.argv) > 1:
        password = sys.argv[1]

        if len(sys.argv) > 2:
            face_img = sys.argv[2]
        else:
            face_img = None
        
        if password != 'skip-extract':
            print("Extracting 7z")
            extract(password)
            print("Extracting 7z Completed")

            image_dir = './static/images/'
            if face_img != None:
                face_path = image_dir + face_img
                face_id = os.path.splitext(os.path.basename(face_path))[0]
                source_face = get_face_single(cv2.imread(face_path))
            else:
                face_id = None
                source_face = None

            extracted_files = len(os.listdir('./static/extracted/'))
            if extracted_files == 0:
                print("Extracting Faces")
                search_files(extract_faces)
                print("Extracting Faces Completed")
            else:
                print('Previous Face Extraction Detected, files: ' + str(extracted_files))

        extracted_files = len(os.listdir('./static/extracted/'))
        if extracted_files != 0:
            print("Extracting Captions")
            #create_captions()
            print("Extracting Captions Completed")
    else:
        print("Invalid Arguments")
