#!/bin/bash
cd "$(dirname "$0")"

PY_SCRIPT=${1:-/app/gradio_app.py}


#You need a webcam attached and you need v4l2loopback enabled
#sudo modprobe v4l2loopback devices=4

docker buildx build -f Dockerfile . -t oneshot-faceswap-web:latest
docker run -it --shm-size=2gb --gpus all \
    -p 127.0.0.1:5000:5000 \
    --mount type=bind,source="$(pwd)/static/static2.bin",target="/tmp/file.7z" \
    --mount type=bind,source=$(pwd)/docker_models_cache/.cache/,target=/root/.cache/ \
    --mount type=bind,source=$(pwd)/docker_models_cache/.insightface/,target=/root/.insightface/ \
    --mount type=bind,source=$(pwd)/docker_models_cache/.superres/,target=/root/.superres/ \
    --mount type=bind,source=$(pwd)/G_latest.pth,target=/app/G_latest.pth \
    --mount type=bind,source=$(pwd)/G_latest_mask.pth,target=/app/G_latest_mask.pth \
    --mount type=bind,source=$(pwd)/backbone.pth,target=/app/backbone.pth \
    --mount type=bind,source=$(pwd)/faces,target=/app/faces/ \
    --mount type=bind,source=$(pwd)/output,target=/app/output/ \
    --mount type=bind,source=$(pwd)/static/videos,target=/app/videos/ \
    --mount type=bind,source=$(pwd)/ghost_models,target=/app/ghost_models/ \
    oneshot-faceswap-web:latest python3 "$PY_SCRIPT"

    #--mount type=bind,source=/mnt/d/TrainingData/img_align_celeba/img_align_celeba,target=/app/celeb \
    #--mount type=bind,source=/mnt/c/Users/Landon/Documents/Code/ghost/output/,target=/app/ghost_models \
