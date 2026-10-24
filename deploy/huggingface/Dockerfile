# Kavach detection backend on a Hugging Face Docker Space (free CPU basic: 2 vCPU, 16 GB RAM).
# Same image as backend/Dockerfile, adapted to Spaces: the container runs as user 1000
# and must listen on port 7860. push_space.py copies backend/ next to this file.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1/libglib2.0-0: runtime libraries opencv-python links against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU torch first, so the pins in requirements.txt are already satisfied and
# pip does not pull the CUDA build from PyPI.
COPY requirements.txt .
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .

# The demo clip is stored in Git LFS (157 MB), too big to upload with the Space.
ARG VIDEO_URL="https://media.githubusercontent.com/media/ritvik302005/KAVACH/main/backend/data/videos/Vehicle%20and%20Surge%20Detection.mp4"
RUN mkdir -p data/videos \
 && curl -fL -o "data/videos/Vehicle and Surge Detection.mp4" "$VIDEO_URL"

# Bake the weights into the image instead of downloading them on every boot.
ENV KAVACH_MODEL=yolov8n.pt
RUN python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt')"

RUN chown -R user:user /home/user/app
USER user
ENV HOME=/home/user \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    PORT=7860 \
    ALLOWED_ORIGINS=https://kavach-surveillance.vercel.app

EXPOSE 7860
CMD ["python", "run.py"]
