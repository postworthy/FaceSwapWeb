
import os
import cv2
os.environ['OMP_NUM_THREADS'] = '1'
import threading
from io import BytesIO
import numpy as np
import time
import torch
import insightface
import onnxruntime
#import tensorflow
from collections import deque
import imageio
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer #https://github.com/postworthy/GFPGAN
import time
import queue
import threading
import ctypes
from ghost import ghost_process_image, ghost_batch_process_image

THREAD_LOCK_FACEANALYSER = threading.Lock()
THREAD_LOCK_FACESWAPPER = threading.Lock()
THREAD_LOCK_UPSAMPLER = threading.Lock()
THREAD_LOCK_UPSAMPLER_FAST = threading.Lock()
THREAD_LOCK_PROCESS = threading.Lock()
UPSAMPLER = None
UPSAMPLER_FAST = None
FACE_SWAPPER = None
FACE_ANALYSER = None

ACTIONS = deque([])

PROVIDERS = onnxruntime.get_available_providers()

if 'TensorrtExecutionProvider' in PROVIDERS:
    PROVIDERS.remove('TensorrtExecutionProvider')

#gpus = tensorflow.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tensorflow.config.experimental.set_memory_growth(gpu, True)

def get_face_analyser():
    global FACE_ANALYSER
    if not FACE_ANALYSER:
        with THREAD_LOCK_FACEANALYSER:
            if not FACE_ANALYSER:
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
                FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_face_swapper():
    global FACE_SWAPPER
    if not FACE_SWAPPER:
        with THREAD_LOCK_FACESWAPPER:
            if not FACE_SWAPPER:
                model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
                print(model_path)
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=PROVIDERS)
    return FACE_SWAPPER

def get_face_single(img_data):
    face = get_face_analyser().get(img_data)
    if face:
        return min(face, key=lambda x: x.bbox[0])
    return None

def get_upsampler(fast=True):
    return get_fast_upsampler() if fast else get_full_upsampler()

def get_full_upsampler():
    #https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    global UPSAMPLER
    if not UPSAMPLER:
        with THREAD_LOCK_UPSAMPLER:
            if not UPSAMPLER:
                bg_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'RealESRGAN_x4plus.pth')
                face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GFPGANv1.4.pth')
                upsampler = RealESRGANer(
                    scale=4,
                    model_path=bg_model_path,
                    dni_weight=None,
                    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    gpu_id=0
                )
                face_upsampler = GFPGANer(
                    model_path=face_model_path,
                    upscale=4,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler
                )
                UPSAMPLER = face_upsampler
    return UPSAMPLER

def get_fast_upsampler():
    #https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    global UPSAMPLER_FAST
    if not UPSAMPLER_FAST:
        with THREAD_LOCK_UPSAMPLER_FAST:
            if not UPSAMPLER_FAST:
                face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GFPGANv1.4.pth')
                face_upsampler = GFPGANer(
                    model_path=face_model_path,
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2
                )
                UPSAMPLER_FAST = face_upsampler
    return UPSAMPLER_FAST

def upsample(image_data, fast=True, has_aligned=False):
    upsampler = get_upsampler(fast)

    with THREAD_LOCK_PROCESS:
        _, _, output = upsampler.enhance(image_data, has_aligned=has_aligned, only_center_face=False, paste_back=True)
        upsampler.cleanup() #Requires using https://github.com/postworthy/GFPGAN
    
    return output

def export_as_gif(frames, fps):
    gif_frames = [frame.astype(np.uint8) for frame in frames]
    # Duplicate frames in reverse order
    #gif_frames += gif_frames[::-1]
    gif_output = BytesIO()
    imageio.mimsave(gif_output, gif_frames, format='gif', duration=1000/fps, loop=0, opt_level=3)
    gif_output.seek(0)

    return gif_output

def export_as_jpg(frames):
    jpg_output = BytesIO()
    imageio.mimsave(jpg_output, [frame.astype(np.uint8) for frame in frames], format='jpg')
    jpg_output.seek(0)

    return jpg_output.getvalue()

def get_fps(id_target):
    video_dir = './static/videos/'
    video_path = video_dir + id_target

    video = cv2.VideoCapture(video_path)
    
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return fps

def push_action(action):
    global ACTIONS
    ACTIONS.append(action)

def get_thread_id(thread):
    # Assumes that the thread is running
    if hasattr(thread, "_thread_id"):
        return thread._thread_id
    for id, t in threading._active.items():
        if t is thread:
            return id
    return None

def raise_exception(thread_id):
    if thread_id == None:
        print("Can't raise Exception for NoneType thread_id")
        return
    
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')


processing_threads = []
processing_threads_lock = threading.Lock()
single_lock = threading.Lock()

def batch_processor(batch_frames, batch_frames_lock, frame_queue, source_images, stop_flag, batch_size=8):
    from util import batch_process_image
    try:
        while True:
            if stop_flag.is_set():
                break
            #print(f"batch_processor running {threading.get_ident()}")
            source_image = source_images[0]
            # Locking the shared resource
            with batch_frames_lock:
                should_process = len(batch_frames) >= batch_size
                if should_process:
                    # Ensure frames_to_process does not exceed batch_==size
                    frames_to_process = batch_frames[:batch_size]
                    # Remove the frames that are about to be processed
                    del batch_frames[:batch_size]  # Retain the remaining unprocessed frames by deleting in place
            
            # Process frames outside of the locked section to not block other thread
            if should_process:
                # Split frames_to_process into chunks of size batch_size and process each chunk
                for i in range(0, len(frames_to_process), batch_size):
                    chunk = frames_to_process[i:i + batch_size]
                    #processed_frames = ghost_batch_process_image(chunk, source_image)
                    processed_frames = batch_process_image(chunk, source_image, False)
                    for processed_frame in processed_frames:
                        frame_queue.put(processed_frame[:, :, ::-1])
            else:
                processed_frames = None
            time.sleep(0.01)  # Sleep briefly to prevent this loop from hogging the CPU
    except SystemExit:
        print("SystemExit Exception Raised")
        pass  
    finally:
        if processed_frames is not None:
            del processed_frames
        print("Thread Exit")
        pass

def process_frames_v2(id_swap, id_target, mode, skip, swaps=[], min_faces=1, upsample_image=False, stop_processing_event=None):  
    global ACTIONS
    global processing_threads    

    try:
        if mode.startswith('single'):
            single_lock.acquire()
            print("single image mode")
        else:
            #clean up previous threads
            with processing_threads_lock:
                still_alive_threads = []
                for thread, flag in processing_threads:
                    if thread.is_alive():
                        tid = get_thread_id(thread)
                        if flag.is_set():
                            raise_exception(tid)
                            print(f"Attempting to Kill: thread-{tid}")
                        else:
                            flag.set()
                            print(f"Stop Flag Set: thread-{tid}")
                        
                        still_alive_threads.append((thread, flag))

                processing_threads = still_alive_threads


        batch_frames = []
        frame_queue = queue.Queue(maxsize=100)
        stop_flag = threading.Event()
        batch_frames_lock = threading.Lock()
        batch_size = 8 if not (mode.startswith('gif') or mode.startswith('single')) else 1
        source_images=[]

        skip_rate=skip

        #face_analyser = get_face_analyser()
        #face_swapper = get_face_swapper()
        
        #if not torch.cuda.is_available():
        #    print("NO CUDA!!!")
        #CUDA_VERSION = torch.version.cuda
        #CUDNN_VERSION = torch.backends.cudnn.version()
        
        image_dir = './static/images/'
        video_dir = './static/videos/'
        video_path = video_dir + id_target

        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_dimension = 500
        scale_width = max_dimension / frame_width
        scale_height = max_dimension / frame_height
        scale_factor = min(scale_width, scale_height)
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)
        max_fps = 30.0
        fps = int(video.get(cv2.CAP_PROP_FPS))
        skip_rate = int(max(1, round(fps / max_fps)))
        fps = int(fps / skip_rate if fps > max_fps else fps)
        frame_duration = 1.0 / fps
        last_frame_time = time.time()
        print(f"FPS: {fps}")
        

        image_path = image_dir + id_swap

        if os.path.exists(image_path+".jpg"):
            image_path = os.path.join(image_path+".jpg")
        elif os.path.exists(image_path+".jpeg"):
            image_path = os.path.join(image_path+".jpeg")
        elif os.path.exists(image_path+".png"):
            image_path = os.path.join(image_path+".png")
        else:
            return
        source_image = cv2.imread(image_path)
        source_images.append(source_image)
        #source_image = upsample(source_image, not mode.startswith('gif'))
        source_face = get_face_single(source_image)
        print(f'Face Dimensions: {(source_face.bbox[2] - source_face.bbox[0])}x{(source_face.bbox[3] - source_face.bbox[1])}')
        
        processing_thread = threading.Thread(target=batch_processor, args=(batch_frames, batch_frames_lock, frame_queue, source_images, stop_flag, batch_size,))
        processing_threads.append((processing_thread, stop_flag))
        processing_thread.start()

        swap_faces = []
        if len(swaps) > 0:
            for swap in swaps:
                if swap != id_swap:
                    swap_path = image_dir + str(swap)
                    if os.path.exists(swap_path+".jpg"):
                        swap_path = os.path.join(swap_path+".jpg")
                    elif os.path.exists(swap_path+".jpeg"):
                        swap_path = os.path.join(swap_path+".jpeg")
                    elif os.path.exists(swap_path+".png"):
                        swap_path = os.path.join(swap_path+".png")
                    else:
                        continue
                    swap_faces.append(get_face_single(cv2.imread(swap_path)))

        frame_index = 0
        ACTIONS = deque([])
        
        break_out = False
        while not break_out and (stop_processing_event == None or not stop_processing_event.is_set()) and not stop_flag.is_set() and video.isOpened():
            if len(ACTIONS) > 0:
                while len(ACTIONS) > 0:
                    action = ACTIONS.popleft()
                    print("action - " + action + "\nremaining in queue " + str(len(ACTIONS)))
                    if action == "pause":
                        print("paused")
                        while len(ACTIONS) == 0 or not ACTIONS.popleft() == "pause":
                            time.sleep(.1)
                        print("unpaused")
                    elif action == "faster":
                        skip_rate += 1
                    elif action == "slower":
                        skip_rate -= 1
                        if skip_rate < 1:
                            skip_rate = 1
                    elif action == "forward":
                        if fps > 0:
                            forward_by = fps*5
                            print("jumping forward by " + str(forward_by))
                            with batch_frames_lock:
                                batch_frames.clear()
                                while not frame_queue.empty():
                                    try:
                                        frame_queue.get_nowait()  # Remove and discard the queue's front item
                                    except queue.Empty:
                                        break  # Just in case another thread emptied the queue
                            for _ in range(0,forward_by):
                                video.read()
                                frame_index = frame_index + 1
                    elif action == "back":
                        if fps > 0: 
                            back_by = fps*5
                            back_to = max(frame_index - back_by, 0)
                            if back_to > 0:
                                frame_index = 0
                                video.release()
                                video = cv2.VideoCapture(video_path)
                                print("jumping back to frame " + str(back_to))
                                if video.isOpened():
                                    for _ in range(0,back_to):
                                        video.read()
                                        frame_index = frame_index + 1
                    elif action == "change-face":
                        image_id = int(os.path.basename(image_path).split('.')[0])
                        images = sorted(os.listdir(image_dir), key=lambda filename: int(filename.split('.')[0]))
                        for i in range(0,len(images)):
                            if image_id == int(images[i].split('.')[0]):
                                try:
                                    image_path = os.path.join(image_dir + images[i+1])
                                except:
                                    image_path = os.path.join(image_dir + images[0])
                        source_image = cv2.imread(image_path)
                        source_images[0] = source_image
                        #source_image = upsample(source_image, not mode.startswith('gif'))
                        source_face = get_face_single(source_image)
                        #source_face = get_face_single(cv2.imread(image_path))
                    elif action == "change-video":
                        with batch_frames_lock:
                                batch_frames.clear()
                                while not frame_queue.empty():
                                    try:
                                        frame_queue.get_nowait()  # Remove and discard the queue's front item
                                    except queue.Empty:
                                        break  # Just in case another thread emptied the queue
                        video_id = int(os.path.basename(video_path).split('.')[0])
                        videos = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
                        for i in range(0,len(videos)):
                            if video_id == int(videos[i].split('.')[0]):
                                try:
                                    video_path = os.path.join(video_dir + videos[i+1])
                                except:
                                    video_path = os.path.join(video_dir + videos[0])
                        
                        video.release()
                        video = cv2.VideoCapture(video_path)
                        if not video.isOpened():
                            return

                continue
            
            for _ in range(0, max(batch_size,fps*skip_rate)):
                # Read a frame from the video
                ret, frame = video.read()
                frame_index = frame_index + 1
                
                #Runs the video faster because we skip frames
                if frame_index % skip_rate: 
                    continue

                if ret:
                    with batch_frames_lock:
                        resized_frame = cv2.resize(frame, (new_width, new_height))
                        batch_frames.append(resized_frame)

                #Runs the video faster because we skip frames
                if frame_index % skip_rate: 
                    continue

            if not ret:
                if mode.startswith('gif') or mode.startswith('single'):
                    break_out = True
                else:
                    #This Loops the stream
                    frame_index = 0
                    video.release()
                    video = cv2.VideoCapture(video_path) 
                    #with batch_frames_lock:
                    #    batch_frames = []
                    if video.isOpened():
                        continue
                    else:
                        break_out = True

            if frame_queue.empty() and len(batch_frames) > 100:
                time.sleep(0.01)

            while not frame_queue.empty():
                result = frame_queue.get()
                    
                # Calculate how long to wait to maintain the desired frame rate
                now = time.time()
                wait_time = last_frame_time + frame_duration - now
                if wait_time > 0:
                    time.sleep(wait_time)  # Pause the loop to maintain the frame rate
                last_frame_time = time.time()  # Reset the last frame time

                #if upsample_image:
                #    result = upsample(result, not mode.startswith('gif'))

                if mode.startswith('gif') or mode == 'single':
                    rgb_frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    yield rgb_frame
                    if mode == 'single':
                        break_out = True
                        break
                else:
                    _, jpeg_frame = cv2.imencode('.jpg', result)
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n\r\n-------------------\r\n\r\n')
    finally:
        batch_frames.clear()
        print("Processing Completed")  
        stop_flag.set()      
        #processing_thread.join()
        video.release()
        print("Exiting: process_frames")

        if mode.startswith('single'):
            single_lock.release()

def process_frames(id_swap, id_target, mode, skip, swaps=[], min_faces=1, upsample_image=False, stop_processing_event=None):  
    global actions  
    
    skip_rate=skip
    
    if not torch.cuda.is_available():
        print("NO CUDA!!!")
    CUDA_VERSION = torch.version.cuda
    CUDNN_VERSION = torch.backends.cudnn.version()
    
    image_dir = './static/images/'
    video_dir = './static/videos/'
    video_path = video_dir + id_target

    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_dimension = 1000
    scale_width = max_dimension / frame_width
    scale_height = max_dimension / frame_height
    scale_factor = min(scale_width, scale_height)
    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)
    max_fps = 60.0
    fps = int(video.get(cv2.CAP_PROP_FPS))
    skip_rate = int(max(1, round(fps / max_fps)))
    fps = int(fps / skip_rate if fps > max_fps else fps)
    frame_duration = 1.0 / fps
    last_frame_time = time.time()
    print(f"FPS: {fps}")
    
    image_path = image_dir + id_swap

    if os.path.exists(image_path+".jpg"):
        image_path = os.path.join(image_path+".jpg")
    elif os.path.exists(image_path+".jpeg"):
        image_path = os.path.join(image_path+".jpeg")
    elif os.path.exists(image_path+".png"):
        image_path = os.path.join(image_path+".png")
    else:
        return
    source_face = get_face_single(cv2.imread(image_path))
    
    swap_faces = []
    if len(swaps) > 0:
        for swap in swaps:
            if swap != id_swap:
                swap_path = image_dir + str(swap)
                if os.path.exists(swap_path+".jpg"):
                    swap_path = os.path.join(swap_path+".jpg")
                elif os.path.exists(swap_path+".jpeg"):
                    swap_path = os.path.join(swap_path+".jpeg")
                elif os.path.exists(swap_path+".png"):
                    swap_path = os.path.join(swap_path+".png")
                else:
                    continue
                swap_faces.append(get_face_single(cv2.imread(swap_path)))

    frame_index = 0
    actions = deque([])

    while (stop_processing_event == None or not stop_processing_event.is_set()) and video.isOpened():
        current_time = time.time()
        elapsed_time = current_time - last_frame_time
        last_frame_time = current_time

        # Adjust skip rate to maintain desired fps
        actual_fps = 1.0 / elapsed_time
        if actual_fps < max_fps:
            skip_rate = max(1, skip_rate - 1)
        elif actual_fps > max_fps:
            skip_rate = min(10,skip_rate + 1)
        
        if len(actions) > 0:
            while len(actions) > 0:
                action = actions.popleft()
                print("action - " + action + "\nremaining in queue " + str(len(actions)))
                if action == "pause":
                    print("paused")
                    while len(actions) == 0 or not actions.popleft() == "pause":
                        time.sleep(.1)
                    print("unpaused")
                elif action == "faster":
                    skip_rate += 1
                elif action == "slower":
                    skip_rate -= 1
                    if skip_rate < 1:
                        skip_rate = 1
                elif action == "forward":
                    if fps > 0:
                        forward_by = fps*5
                        print("jumping forward by " + str(forward_by))
                        for x in range(0,forward_by):
                            video.read()
                            frame_index = frame_index + 1
                elif action == "back":
                    if fps > 0: 
                        back_by = fps*5
                        back_to = frame_index - back_by if frame_index - back_by >= 0 else frame_index
                        if back_to > 0:
                            frame_index = 0
                            video.release()
                            video = cv2.VideoCapture(video_path)
                            print("jumping back to frame " + str(back_to))
                            if video.isOpened():
                                for x in range(0,back_to):
                                    video.read()
                                    frame_index = frame_index + 1
                elif action == "change-face":
                    image_id = int(os.path.basename(image_path).split('.')[0])
                    images = sorted(os.listdir(image_dir), key=lambda filename: int(filename.split('.')[0]))
                    for i in range(0,len(images)):
                        if image_id == int(images[i].split('.')[0]):
                            try:
                                image_path = os.path.join(image_dir + images[i+1])
                            except:
                                image_path = os.path.join(image_dir + images[0])
                    
                    source_face = get_face_single(cv2.imread(image_path))
                elif action == "change-video":
                    video_id = int(os.path.basename(video_path).split('.')[0])
                    videos = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
                    for i in range(0,len(videos)):
                        if video_id == int(videos[i].split('.')[0]):
                            try:
                                video_path = os.path.join(video_dir + videos[i+1])
                            except:
                                video_path = os.path.join(video_dir + videos[0])
                    
                    video.release()
                    video = cv2.VideoCapture(video_path)
                    if not video.isOpened():
                        return

            continue

        # Read a frame from the video
        ret, frame = video.read()
        frame_index = frame_index + 1

        #Runs the video faster because we skip frames
        if frame_index % skip_rate: 
            continue

        if not ret:
            if mode.startswith('gif') or mode.startswith('single'):
                break
            else:
                #This Loops the stream
                frame_index = 0
                video.release()
                video = cv2.VideoCapture(video_path) 
                if video.isOpened():
                    continue
                else:
                    break

        resized_frame = cv2.resize(frame, (new_width, new_height))
        frame = resized_frame

        faces = sorted(get_face_analyser().get(frame), key=lambda x: x.bbox[0])
        if faces:
            if min_faces > 1 and len(faces) < min_faces:
                continue
            else:
                for i, face in enumerate(faces):
                    if i == 0 or len(swap_faces) == 0:
                        result = get_face_swapper().get(frame, face, source_face, paste_back=True)
                        frame = result
                    else:
                        swap_face = swap_faces[i % len(swap_faces)]
                        result = get_face_swapper().get(frame, face, swap_face, paste_back=True)
                        frame = result
        elif mode == 'single' or mode == 'gif-skip':
            continue
        else:
            time.sleep(max(0, frame_duration - elapsed_time))
            result = frame
        
        if upsample_image:
            with THREAD_LOCK_PROCESS:
                result = upsample(result, not mode.startswith('gif'))


        _, jpeg_frame = cv2.imencode('.jpg', result)

        if mode.startswith('gif'):
            decoded_frame = cv2.imdecode(np.frombuffer(jpeg_frame, np.uint8), cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
            yield(rgb_frame)
        elif mode == 'single':
            decoded_frame = cv2.imdecode(np.frombuffer(jpeg_frame, np.uint8), cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
            yield(rgb_frame)
            break
        else:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n\r\n-------------------\r\n\r\n')


    print("Processing Completed")        
    # Release the resources
    video.release()