#Requires FaceSwap Pipeline to be build local
FROM faceswap-pipeline:latest

#WORKDIR /app
#ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y xz-utils ffmpeg && rm -rf /var/lib/apt/lists/*
#libcublas-11-8 libcublas-dev-11-8
#RUN pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install onnxruntime-gpu
RUN pip3 install ftfy regex tqdm
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install simple-websocket
RUN mkdir /app/templates
RUN mkdir /app/static
RUN mkdir /app/static/images
RUN mkdir /app/static/videos
RUN mkdir /app/static/cache
RUN mkdir -p /root/.insightface/models/
ADD ./requirements.txt /app
RUN pip3 install -r requirements.txt
#ADD inswapper_128.onnx /app/ 
RUN ln -s /root/.insightface/models/inswapper_128.onnx /app/inswapper_128.onnx

#RUN pip3 install basicsr 
#RUN pip3 install facexlib

#RUN git clone https://github.com/postworthy/GFPGAN.git
#RUN cd GFPGAN && pip install -r requirements.txt && python3 setup.py develop
#RUN ln -s GFPGAN/gfpgan .

#RUN git clone https://github.com/xinntao/Real-ESRGAN.git
#RUN cd Real-ESRGAN && pip install -r requirements.txt && python3 setup.py develop
#RUN ln -s Real-ESRGAN/realesrgan .
#RUN mkdir -p /app/gfpgan/weights/

#RUN ln -s /root/.superres/realesr-general-x4v3.pth /app/realesr-general-x4v3.pth
#RUN ln -s /root/.superres/RealESRGAN_x4plus.pth /app/RealESRGAN_x4plus.pth
#RUN ln -s /root/.superres/GFPGANv1.4.pth /app/GFPGANv1.4.pth
#RUN ln -s /root/.superres/detection_Resnet50_Final.pth /app/gfpgan/weights/detection_Resnet50_Final.pth
#RUN ln -s /root/.superres/parsing_parsenet.pth /app/gfpgan/weights/parsing_parsenet.pth

#RUN python3 -c "import torch; import insightface; import onnxruntime; import tensorflow; PROVIDERS = onnxruntime.get_available_providers(); [PROVIDERS.remove(provider) for provider in PROVIDERS if provider == 'TensorrtExecutionProvider']; insightface.app.FaceAnalysis(name='buffalo_l', providers=PROVIDERS)" || true
#RUN python3 -c "import os, torch; from torchvision.transforms import Compose, Resize, ToTensor; from PIL import Image; from clip import clip; device = 'cuda' if torch.cuda.is_available() else 'cpu'; model, preprocess = clip.load('ViT-B/32', device=device)"

#ADD ./interrogate.py /app
#RUN python3 -c "from interrogate import InterrogateModels;interrogator = InterrogateModels('interrogate');interrogator.load();interrogator.categories()"

#RUN wget -O /app/realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
#RUN wget -O /app/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
#RUN wget -O /app/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
#RUN wget -O /app/gfpgan/weights/detection_Resnet50_Final.pth https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
#RUN wget -O /app/gfpgan/weights/parsing_parsenet.pth https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

#RUN cd GFPGAN && git pull

#RUN apt-get install -y libarchive-dev libfuse-dev
#RUN apt-get install -y pkg-config
#RUN git clone https://github.com/google/fuse-archive.git
#RUN cd fuse-archive && make && make install

#PATCH https://github.com/XPixelGroup/BasicSR/pull/624/files
#RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py
#END PATCH

ADD ./templates/*.html /app/templates/
ADD ./app.py /app
ADD ./app2.py /app
ADD ./fsw_util.py /app
ADD ./face-extraction.py /app

RUN pip3 install python-telegram-bot
RUN pip3 install gradio

ADD ./bot.py /app
ADD ./faceswap_mp4.py /app
ADD ./gradio_app.py /app

EXPOSE 5000

#CMD ["python3", "app2.py"]
#CMD ["python3", "faceswap_mp4.py"]
CMD ["python3", "gradio_app.py"]