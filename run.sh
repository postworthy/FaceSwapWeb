#!/bin/bash
cd "$(dirname "$0")"

docker buildx build -f Dockerfile . -t oneshot-faceswap-web:latest
docker run -it --shm-size=2gb --gpus all \
    -p 5555:5000 \
    --mount type=bind,source="$(pwd)/static/static2.bin",target="/tmp/file.7z" \
    --mount type=bind,source=$(pwd)/docker_models_cache/.cache/,target=/root/.cache/ \
    --mount type=bind,source=$(pwd)/docker_models_cache/.insightface/,target=/root/.insightface/ \
    --mount type=bind,source=$(pwd)/docker_models_cache/.superres/,target=/root/.superres/ \
    --mount type=bind,source=$(pwd)/G_latest.pth,target=/app/G_latest.pth \
    --mount type=bind,source=$(pwd)/G_latest_mask.pth,target=/app/G_latest_mask.pth \
    --mount type=bind,source=$(pwd)/backbone.pth,target=/app/backbone.pth \
    --mount type=bind,source=$(pwd)/faces,target=/app/faces/ \
    --mount type=bind,source=$(pwd)/output,target=/app/output/ \
    oneshot-faceswap-web:latest
