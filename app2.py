from flask import Flask, render_template, send_file, make_response, Response, request, session, redirect, url_for, abort
import os
import cv2
os.environ['OMP_NUM_THREADS'] = '1'
import threading
from PIL import Image
from io import BytesIO
import datetime
from flask_socketio import SocketIO, send
from collections import deque
import py7zr
import secrets
import string
from fsw_util import push_action, export_as_gif, export_as_jpg, get_face_single, get_face_swapper, process_frames, process_frames_v2, get_fps, get_face_analyser
import numpy as np

app = Flask(__name__)

use_ghost = True

if use_ghost:
    _process_frames = process_frames_v2
else:
    _process_frames = process_frames

THREAD_LOCK_UPLOAD = threading.Lock()

socketio = SocketIO(app)
zip_path = "/tmp/file.7z"
zip_pass = ""

@app.before_request
def authenticate():
    if request.endpoint not in ['login', 'logout'] and 'authenticated' not in session:
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
            print("Unzip Fail!")
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
    image_files = sorted((file for file in os.listdir(image_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png'))), key=lambda filename: int(filename.split('.')[0]))
    return render_template('index.html', images=image_files)

@app.route('/base/images/<id>')
def base_images(id):
    image_dir = './static/images/'
    source_face = get_face_single(cv2.imread(image_dir + id)[:,:,::-1])
    #source_face = cv2.imread(image_dir + id)
    #source_face = Image.open(image_dir + id)
    #print(source_face.gender)
    
    #frame = cv2.imread('./static/base.jpg')
    #face = get_face_single(frame)
    
    #result = get_face_swapper().get(frame, face, source_face, paste_back=True)
    #jpg_img = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    #image = Image.fromarray(jpg_img)
    
    #image = Image.fromarray(source_face)
    image = source_face
    image_stream = BytesIO()
    image.save(image_stream, format='PNG')
    image_stream.seek(0)
    
    image_response = make_response(send_file(image_stream, mimetype='image/png'))
    expires_at = datetime.datetime.now() + datetime.timedelta(hours=24)
    image_response.headers['Cache-Control'] = 'public, max-age=86400'
    image_response.headers['Expires'] = expires_at.strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    return image_response

@app.route('/streams/<id>')
def streams(id):
    upsample = request.args.get("upsample", default=0, type=int)
    video_dir = './static/videos/'
    videos = sorted((file for file in os.listdir(video_dir) if file.lower().endswith(('.mp4', '.gif'))), key=lambda filename: int(filename.split('.')[0]))
    return render_template('streams.html', videos=videos, id=id, upsample=str(upsample))

@app.route('/grid/<columns>')
def stream_grid(columns):
    swaps = request.args.getlist("swaps")
    videos = request.args.getlist("videos")
    skip = request.args.get("skip", default=1, type=int)

    pairs = [(id_swap, id_video) for id_swap in swaps for id_video in videos]

    return render_template('grid.html', pairs=pairs, columns=columns, skip=skip)

@app.route('/stream/<id_swap>/<id_target>')
def stream_video(id_swap, id_target):
    upsample = request.args.get("upsample", default=0, type=int)
    swaps = '&swaps='.join(map(str, request.args.getlist("swaps")))
    return render_template('streaming.html', id_swap=id_swap, id_target=id_target, upsample=str(upsample), swaps=swaps)

stop_processing_event = None

@app.route('/stream/data/<id_swap>/<id_target>')
def stream_video_data(id_swap, id_target):
    global stop_processing_event

    swaps = request.args.getlist("swaps")
    mode = request.args.get("mode", default="full", type=str)
    skip = request.args.get("skip", default=1, type=int)
    min_faces = request.args.get("min", default=1, type=int)
    upsample = request.args.get("upsample", default=0, type=int)
    if mode.startswith('gif'):
        frames = [frame for frame in _process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0, use_ghost=use_ghost)]
        gif_output = export_as_gif(frames, get_fps(id_target))
        return send_file(gif_output, mimetype='image/gif', as_attachment=True, download_name='output.gif')
    if mode.startswith('single'):
        existing_file_name = id_swap + "_" + id_target + ".jpg"
        existing_file_path = './static/cache/'
        if not os.path.exists(existing_file_path + existing_file_name):
            with THREAD_LOCK_UPLOAD:
                file_name = id_swap + "_" + id_target + ".jpg"
                file_path = 'cache/'
                try:
                    with open(existing_file_path + existing_file_name, 'xb') as file:
                        file.write(export_as_jpg([frame for frame in _process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0, use_ghost=use_ghost)]))
                except Exception as e: 
                    print(e)
                    if os.path.exists(existing_file_path + existing_file_name):
                        os.remove(existing_file_path + existing_file_name)
                    return abort(500)

                with py7zr.SevenZipFile(zip_path, 'a', password=zip_pass) as archive:
                    print(file_path, file_name)
                    archive.write(existing_file_path + existing_file_name, file_path + file_name)
        
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
        if stop_processing_event != None:
            stop_processing_event.set()        
        stop_processing_event = threading.Event()
        try:
            
            stop_processing_event.clear()
            def generate():
                try:
                    for frame in _process_frames(id_swap, id_target, mode, skip, swaps, min_faces, upsample > 0, stop_processing_event, use_ghost=use_ghost) :
                        yield frame
                except GeneratorExit:  # Catch the GeneratorExit when client disconnects
                    stop_processing_event.set()
                finally:
                    stop_processing_event.set()  # Ensure the stop signal is set if the generator ends normally


            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame') 
        except Exception as e:
            stop_processing_event.set()  # Make sure to signal to stop on error
            raise e
        
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        with THREAD_LOCK_UPLOAD:
            file = request.files['file']
            video_dir = './static/videos/'
            image_dir = './static/images/'

            _, file_extension = os.path.splitext(file.filename)
            if file_extension in ['.gif', '.mp4']:
                files = sorted(os.listdir(video_dir), key=lambda filename: int(filename.split('.')[0]))
                file_id = str(len(files) + 1)
                file_name = file_id + file_extension
                file_path = 'videos/'
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                files = sorted(os.listdir(image_dir), key=lambda filename: int(filename.split('.')[0]))
                file_id = str(len(files) + 1)
                file_name = file_id + file_extension
                file_path = 'images/'
            else:
                abort(500)

            file.save('static/' + file_path + file_name)

            with py7zr.SevenZipFile(zip_path, 'a', password=zip_pass) as archive:
                archive.write(f'static/{file_path}{file_name}', f'{file_path}{file_name}')

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
    video_files = sorted((file for file in os.listdir(video_dir) if file.lower().endswith(('.mp4', '.gif'))), key=lambda filename: int(filename.split('.')[0]))
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
    push_action(message)

@socketio.on('disconnect')
def handle_disconnect():
    if stop_processing_event != None:
        stop_processing_event.set()

if __name__ == '__main__':
    app.secret_key = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(50))
    socketio.run(app, host='0.0.0.0', port=5000)