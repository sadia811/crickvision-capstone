FROM --platform=linux/amd64 python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ffmpeg: MP4/H.264 output; shared libraries: OpenCV/PyTorch runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-docker.txt /app/requirements-docker.txt

RUN python -m pip install --upgrade 'pip<27' 'setuptools<82' wheel

# A matching CPU-only Torch/MMCV pair. DO NOT replace MMCV with mmcv-lite.
RUN python -m pip install 'numpy==1.26.4'
RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.1.0+cpu' 'torchvision==0.16.0+cpu'
RUN python -m pip install --no-deps \
    https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl

RUN python -m pip install -r /app/requirements-docker.txt

# Exact MMPose version used for training. --no-deps bypasses the broken
# legacy chumpy sdist build; other runtime dependencies are listed explicitly.
RUN python -m pip install --no-deps 'mmpose==1.3.2'

# Fail the build before deployment if compiled MMCV or MMPose imports break.
RUN python -c "import torch, mmcv, mmengine, mmdet, mmpose; from mmcv.ops import roi_align; from mmpose.apis import init_model, inference_topdown; print('MMPose/MMDTE/MMCV import check OK:', torch.__version__, mmcv.__version__, mmpose.__version__)"

COPY . /app/
EXPOSE 10000

# Render provides $PORT; Streamlit binds to all interfaces for public access.
CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-10000} --server.headless true --browser.gatherUsageStats false"]
