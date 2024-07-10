from flask import Flask, render_template, send_file, make_response, Response, request, session, redirect, url_for, abort
import os
import cv2
os.environ['OMP_NUM_THREADS'] = '1'
import threading
from PIL import Image
from io import BytesIO
import datetime
import numpy as np
import time
import torch
import insightface
import onnxruntime
import tensorflow
import re
import tempfile
from flask_socketio import SocketIO, send
from collections import deque
import py7zr
import secrets
import string
import imageio
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

app = Flask(__name__)

THREAD_LOCK_UPLOAD = threading.Lock()
THREAD_LOCK_FACE = threading.Lock()
THREAD_LOCK_UPSAMPLER = threading.Lock()
THREAD_LOCK_UPSAMPLER_FAST = threading.Lock()
THREAD_LOCK_PROCESS = threading.Lock()
UPSAMPLER = None
UPSAMPLER_FAST = None
FACE_SWAPPER = None
FACE_ANALYSER = None

socketio = SocketIO(app)
actions = deque([])
zip_path = "/tmp/static.7z"
zip_pass = ""

PROVIDERS = onnxruntime.get_available_providers()

if 'TensorrtExecutionProvider' in PROVIDERS:
    PROVIDERS.remove('TensorrtExecutionProvider')

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=PROVIDERS)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_face_swapper():
    global FACE_SWAPPER
    with THREAD_LOCK_FACE:
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
            print(model_path)
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=PROVIDERS)
    return FACE_SWAPPER

def get_face_single(img_data):
    face = get_face_analyser().get(img_data)
    try:
        return sorted(face, key=lambda x: x.bbox[0])[0]
    except IndexError:
        return None

def get_upsampler(fast=True):
    if fast:
        return get_fast_upsampler()
    else:
        return get_full_upsampler()
    
def get_full_upsampler():
    #https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
    global UPSAMPLER
    with THREAD_LOCK_UPSAMPLER:
        if UPSAMPLER is None:
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
                gpu_id=0)
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
    with THREAD_LOCK_UPSAMPLER_FAST:
        if UPSAMPLER is None:
            face_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'GFPGANv1.4.pth')
            face_upsampler = GFPGANer(
                model_path=face_model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2
            )
            UPSAMPLER_FAST = face_upsampler
    return UPSAMPLER_FAST

def upsample(image_data, fast=True):
    upsampler = get_upsampler(fast)

    #output, _ = upsampler.enhance(image_data, outscale=4)
    _, _, output = upsampler.enhance(image_data, has_aligned=False, only_center_face=False, paste_back=True)
    return output


def export_as_gif(frames, fps):
    gif_frames = [frame.astype(np.uint8) for frame in frames]
    # Duplicate frames in reverse order
    #gif_frames += gif_frames[::-1]
    gif_output = BytesIO()
    imageio.mimsave(gif_output, gif_frames, format='gif', duration=1000/fps, loop=0)
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
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return fps

@app.before_request
def authenticate():
    if request.endpoint not in ['login', 'logout'] and not session.get('authenticated'):
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    global zip_pass
    if request.method == 'POST':
        password = request.form['password']
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
            return redirect(url_for('login'))
        else:
            # Password is correct, grant access
            session['authenticated'] = True
            return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    image_dir = './static/images'
    image_files = sorted(os.listdir(image_dir), key=lambda filename: int(filename.split('.')[0]))
    images = []
    for file in image_files:
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            images.append(file)
    return render_template('index.html', images=images)

@app.route('/base/images/<id>')
def base_images(id):
    image_dir = './static/images/'
    source_face = get_face_single(cv2.imread(image_dir + id))
    print(source_face.gender)
    #if source_face.gender == 1:
    frame = cv2.imread('./static/base.jpg')
    face = get_face_single(frame)
    #else:
    #    frame = cv2.imread('./static/base2.jpg')
    #    face = get_face_single(frame)
    result = get_face_swapper().get(frame, face, source_face, paste_back=True)
    jpg_img = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    image = Image.fromarray(jpg_img)
    #image = Image.fromarray(result)
    image_stream = BytesIO()
    image.save(image_stream, format='PNG')
    image_stream.seek(0)
    image_response = make_response(send_file(image_stream, mimetype='image/png'))
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=24)
    image_response.headers['Cache-Control'] = 'public, max-age=86400'
    image_response.headers['Expires'] = expires_at.strftime('%a, %d %b %Y %H:%M:%S GMT')
    return image_response

#@app.route('/video/<id>')
#def video(id):
#    video_dir = './static/videos/' + id
#    video_files = os.listdir(video_dir)
#    videos = []
#    for file in video_files:
#        if file.endswith('.mp4'):
#            videos.append(file)
#    return render_template('video.html', videos=videos, id=id)

@app.route('/streams/<id>')
def streams(id):
    upsample = request.args.get("upsample", default=0, type=int)
    video_dir = './static/videos/'
    video_files = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
    videos = []
    for file in video_files:
        if file.endswith('.mp4') or file.endswith('.gif'):
            videos.append(file)
    return render_template('streams.html', videos=videos, id=id, upsample=str(upsample))

@app.route('/grid/<columns>')
#http://127.0.0.1:5000/grid/3?swaps=2&swaps=9&swaps=11&videos=32.mp4&skip=4
def stream_grid(columns):
    swaps = request.args.getlist("swaps")
    videos = request.args.getlist("videos")
    skip = request.args.get("skip", default=1, type=int)

    pairs = []

    for id_swap in swaps:
        for id_video in videos:
            pairs.append((id_swap, id_video))

    return render_template('grid.html', pairs=pairs, columns=columns, skip=skip)

@app.route('/stream/<id_swap>/<id_target>')
def stream_video(id_swap, id_target):
    upsample = request.args.get("upsample", default=0, type=int)
    swaps = request.args.getlist("swaps")
    return render_template('streaming.html', id_swap=id_swap, id_target=id_target, upsample=str(upsample), swaps='&swaps='.join(map(str, swaps)))

@app.route('/stream/data/<id_swap>/<id_target>')
def stream_video_data(id_swap, id_target):
    swaps = request.args.getlist("swaps")
    mode = request.args.get("mode", default="full", type=str)
    skip = request.args.get("skip", default=1, type=int)
    min_faces = request.args.get("min", default=1, type=int)
    upsample = request.args.get("upsample", default=0, type=int)
    if mode.startswith('gif'):
        frames = [frame for frame in process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0)]
        gif_output = export_as_gif(frames, get_fps(id_target))
        return send_file(gif_output, mimetype='image/gif', as_attachment=True, download_name='output.gif')
    if mode.startswith('single'):
        #old code can be removed when the other code works
        #expires_at = datetime.datetime.now() + datetime.timedelta(hours=24)
        #headers = { 
        #    'Cache-Control': 'public, max-age=86400',
        #    'Expires' : expires_at.strftime('%a, %d %b %Y %H:%M:%S GMT')
        #    }
        #return Response(process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0), headers=headers, mimetype='multipart/x-mixed-replace; boundary=frame') 
    
        existing_file_name = id_swap + "_" + id_target + ".jpg"
        existing_file_path = './static/cache/'
        if not os.path.exists(existing_file_path + existing_file_name):
            with THREAD_LOCK_UPLOAD:
                file_name = id_swap + "_" + id_target + ".jpg"
                file_path = 'cache/'
                with open('./static/' + file_path + file_name, 'xb') as file:
                    file.write(export_as_jpg([frame for frame in process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0)]))

                with py7zr.SevenZipFile(zip_path, 'a', password=zip_pass) as archive:
                    print(file_path, file_name)
                    archive.write('static/' + file_path + file_name, file_path + file_name)
        
        if os.path.exists(existing_file_path + existing_file_name):
            expires_at = datetime.datetime.now() + datetime.timedelta(hours=24)
            headers = {
                'Cache-Control': 'public, max-age=86400',
                'Expires': expires_at.strftime('%a, %d %b %Y %H:%M:%S GMT')
            }
            response = make_response(send_file(existing_file_path + existing_file_name, mimetype='image/jpeg'))
            response.headers = headers
            return response
        else:
            abort(404)
    else:
        return Response(process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0), mimetype='multipart/x-mixed-replace; boundary=frame') 

def process_frames(id_swap, id_target, mode, skip, swaps=[], min_faces=1, upsample_image=False):  
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
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
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

    while video.isOpened():
        
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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        with THREAD_LOCK_UPLOAD:
            file = request.files['file']

            video_dir = './static/videos/'
            image_dir = './static/images/'

            _, file_extension = os.path.splitext(file.filename)
            if file_extension == '.gif' or file_extension == '.mp4':
                files = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
                file_id = str(len(files) + 1)
                file_name = file_id + file_extension
                file_path = 'videos/' 
                file.save('static/' + file_path + file_name)
            elif file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
                files = sorted(os.listdir(image_dir), key=lambda filename: int(filename.split('.')[0]))
                file_id = str(len(files) + 1)
                file_name = file_id + file_extension
                file_path = 'images/'
                file.save('static/' + file_path + file_name)
            else:
                abort(500)

            with py7zr.SevenZipFile(zip_path, 'a', password=zip_pass) as archive:
                print(file_path, file_name)
                archive.write('static/' + file_path + file_name, file_path + file_name)

            return '', 200
    else:
        return render_template('upload.html')

@app.route('/groups')
def groups():
    swaps = request.args.getlist("swaps")

    if len(swaps) == 0:
        abort(500)
    else:
        id=str(swaps[0])

    video_dir = './static/videos/'
    video_files = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
    videos = []
    
    for file in video_files:
        if file.endswith('.mp4') or file.endswith('.gif'):
            print(file)
            video = cv2.VideoCapture(video_dir + file)
            while video.isOpened():
                ret, frame = video.read()
                frames_checked = 0
                if not ret:
                    break
                faces = get_face_analyser().get(frame) #get_face_single(frame)
                if faces:
                    if len(faces) > 1:
                        print(str(len(faces)) + " - " + file)
                        frames_checked = frames_checked + 1
                        videos.append(file)
                        break
                
                if frames_checked > 20:
                    break

            video.release()

    response = make_response(render_template('groups.html', videos=videos, id=id, swaps='&swaps='.join(map(str, swaps))))
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=24)
    response.headers['Cache-Control'] = 'public, max-age=86400'
    response.headers['Expires'] = expires_at.strftime('%a, %d %b %Y %H:%M:%S GMT')
    return response

@socketio.on('message')
def handle_message(message):
    print('Received message:', message)
    global actions
    actions.append(message)

def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )


if __name__ == '__main__':
    app.secret_key = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(50))
    #app.run()
    #if is_docker():
        #password=password=os.environ['7zpass']
        #with py7zr.SevenZipFile(zip_path, mode='r', password=password) as archive: 
        #    archive.extractall(path="./static/")


    socketio.run(app, host='0.0.0.0', port=5000)