#!/bin/bash
cd "$(dirname "$0")"

#sudo modprobe v4l2loopback

docker buildx build -f Dockerfile-fakewebcam . -t oneshot-faceswap-cam:latest
docker run -it --shm-size=2gb --gpus all \
    -p 5555:5000 \
    --mount type=bind,source="$(pwd)/static/images/",target="/app/static/images/" \
    --mount type=bind,source="$(pwd)/CURRENT_IMAGE_INDEX",target="/app/CURRENT_IMAGE_INDEX" \
    --mount type=bind,source=$(pwd)/docker_models_cache/.cache/,target=/root/.cache/ \
    --mount type=bind,source=$(pwd)/docker_models_cache/.insightface/,target=/root/.insightface/ \
    --mount type=bind,source=$(pwd)/docker_models_cache/.superres/,target=/root/.superres/ \
    --mount type=bind,source=$(pwd)/G_latest.pth,target=/app/G_latest.pth \
    --mount type=bind,source=$(pwd)/G_latest_mask.pth,target=/app/G_latest_mask.pth \
    --mount type=bind,source=$(pwd)/backbone.pth,target=/app/backbone.pth \
    --mount type=bind,source=$(pwd)/faces,target=/app/faces/ \
    --mount type=bind,source=$(pwd)/output,target=/app/output/ \
    --mount type=bind,source=/dev/video0,target=/dev/video0 \
    --mount type=bind,source=/dev/video4,target=/dev/video1 \
    --privileged \
    oneshot-faceswap-cam:latest
