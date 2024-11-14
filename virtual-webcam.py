import os
os.environ['OMP_NUM_THREADS'] = '1'
import cv2
import pyfakewebcam
from fsw_util import get_face_analyser, get_face_swapper, get_face_single, upsample
from ghost import ghost_batch_process_image
import time

invert_input = True
stop_processing_event = None
enable_upsample = False
use_ghost = True
swap_on_no_face = True
real_camera=cv2.VideoCapture('/dev/video0')
frame_width=int(real_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(real_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam Resolution {frame_width}x{frame_height}")
camera = pyfakewebcam.FakeWebcam('/dev/video1', 640, 480)
min_faces=1
image_dir = './static/images/'
shared_file = '/app/CURRENT_IMAGE_INDEX'

# List all available images
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not images:
    print("No images found in the directory. Exiting...")
    exit()
else:
    images.append("NONE")

def load_image(image_index):
    """Load the image at the specified index."""
    if "NONE" in images[image_index]:
        return None, None
    image_path = os.path.join(image_dir, images[image_index])
    return cv2.imread(image_path), image_path

def get_current_image_index():
    """Get the current image index from the shared file."""
    try:
        with open(shared_file, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 1


# Initial image load
current_image_index = get_current_image_index()
current_image_index = (current_image_index + 1) % len(images)
source_image, image_path = load_image(current_image_index)

print(f"Loaded image: {image_path}")

#if os.path.exists(image_path+".jpg"):
#    image_path = os.path.join(image_path+".jpg")
#elif os.path.exists(image_path+".jpeg"):
#    image_path = os.path.join(image_path+".jpeg")
#elif os.path.exists(image_path+".png"):
#    image_path = os.path.join(image_path+".png")
#else:
#    print("No source image found, exiting...")
#    exit()

if source_image is not None:
    source_image = cv2.imread(image_path)

    if not use_ghost:
        source_face = get_face_single(source_image)
    else:
        source_face = None


print("Streaming started. Press Ctrl+C to stop.")

try:
    frame_counter = 0
    swapped = False
    while True:
        
        # Capture frame-by-frame
        ret, frame = real_camera.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        
        if invert_input == True:
            # Invert the frame
            frame = cv2.flip(frame, 0)

         # Increment the frame counter
        frame_counter += 1

        # Skip every N frame
        #if frame_counter % 2 == 0:
        #    continue

        if source_image is None:
            frame = frame[:,:,::-1]
            camera.schedule_frame(frame)
            swapped = True
        else:
            if use_ghost:
                frame = ghost_batch_process_image([frame], source_image)[0]
                camera.schedule_frame(frame)
                swapped = True
            else:
                try:
                    faces = sorted(get_face_analyser().get(frame), key=lambda x: x.bbox[0])
                    if faces:
                        if min_faces > 1 and len(faces) < min_faces:
                            continue
                        else:
                            for i, face in enumerate(faces):
                                result = get_face_swapper().get(frame, face, source_face, paste_back=True)
                                frame = result[:,:,::-1]
                                if enable_upsample:
                                    frame = upsample(frame)
                                    height, width = frame.shape[:2]
                                    new_width = width // 2
                                    new_height = height // 2
                                    new_dimensions = (new_width, new_height)
                                    frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)

                                camera.schedule_frame(frame)
                                swapped = True
                    else:
                        camera.schedule_frame(frame)
                                    
                except Exception as e:
                    raise e
            
        if (swapped and swap_on_no_face and get_face_single(frame) == None):# or current_image_index != get_current_image_index():
            swapped = False
            if swap_on_no_face:
                current_image_index += 1
            else:
                current_image_index = get_current_image_index()
            
            # Change to the next image
            if current_image_index + 1 > len(images):
                current_image_index = 0
            
            source_image, image_path = load_image(current_image_index)
            print(f"Loaded image: {image_path}")
            if source_image is not None:
                if not use_ghost:
                    source_face = get_face_single(source_image)
                else:
                    source_face = None

             # Sleep for 1 second after a swap
            time.sleep(.5)

except KeyboardInterrupt:
    print("Streaming stopped.")