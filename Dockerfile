FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Build args for non-root runtime user
ARG UID=1000
ARG GID=1000
ARG USERNAME=appuser
ARG DOWNLOAD_WEIGHTS=1

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt

# Install system dependencies required for preprocessing, training and inference.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        ffmpeg \
        git \
        wget \
        curl \
        ca-certificates \
        build-essential \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    python -m pip install --no-cache-dir "pip<24.1" setuptools wheel

# Install PyTorch CUDA 11.8 wheels first as instructed in README.
RUN python -m pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies from requirements, excluding the broken editable fairseq URL.
RUN grep -v "^-e git+https://github.com/pytorch/fairseq" /tmp/requirements.txt > /tmp/requirements-clean.txt && \
    python -m pip install --no-cache-dir -r /tmp/requirements-clean.txt

# Copy repository files into the image.
COPY . /workspace

# Install local fairseq submodule in editable mode.
RUN python -m pip install --no-cache-dir --editable /workspace/av_hubert/fairseq

# ============================================================================
# SETUP RETINAFACE & MODIFICATIONS
# ============================================================================
RUN cp -r modification/retinaface preprocessing/face-alignment/face_alignment/detection && \
    cp modification/landmark_extract.py preprocessing/face-alignment/

# ============================================================================
# DOWNLOAD MODEL CHECKPOINTS
# ============================================================================
# gdown is more reliable for Google Drive in non-interactive builds.
RUN mkdir -p checkpoints && \
    if [ "${DOWNLOAD_WEIGHTS}" = "1" ]; then \
        python -m pip install --no-cache-dir gdown && \
        [ -f checkpoints/Resnet50_Final.pth ] || gdown "https://drive.google.com/uc?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1" -O checkpoints/Resnet50_Final.pth && \
        [ -f checkpoints/large_vox_iter5.pt ] || wget -q --show-progress -O checkpoints/large_vox_iter5.pt https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt; \
    else \
        echo "Skipping weight download (DOWNLOAD_WEIGHTS=${DOWNLOAD_WEIGHTS})"; \
    fi

# Create a non-root runtime user.
RUN if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" "${USERNAME}"; fi && \
    if ! getent passwd "${UID}" >/dev/null; then \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    fi && \
    chown -R "${UID}:${GID}" /workspace

ENV HOME=/home/${USERNAME}
USER ${USERNAME}

# ============================================================================
# ENTRYPOINT
# ============================================================================
CMD ["bash"]

