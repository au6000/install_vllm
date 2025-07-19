# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="90" \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=0

# 最小限インストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git curl build-essential python3-pip && \
    rm -rf /var/lib/apt/lists/*

# PyTorch stack
RUN pip install -U pip setuptools && \
    pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
      --index-url https://download.pytorch.org/whl/cu124 && \
    pip cache purge

# Core LLM tools
RUN pip install --no-cache-dir \
      triton==3.2.0 \
      transformers==0.22.0 \
      datasets==2.21.0 \
      deepspeed==0.17.2 \
      accelerate==0.29.2 \
      evaluate wandb tiktoken \
      bitsandbytes==0.43.1 \
      xformers==0.0.33

# Flash-Attn 3
RUN git clone https://github.com/Dao-AILab/flash-attention.git /opt/flash-attn && \
    cd /opt/flash-attn && \
    git checkout v3.0.0 && \
    python setup.py install && \
    rm -rf /opt/flash-attn

# Unsloth
RUN pip install --no-cache-dir "unsloth[gpu]"==0.6.0

# Smoke test
RUN python -c "import unsloth; print('Unsloth OK:', unsloth.utils.get_compiled())"

WORKDIR /workspace
COPY ds_zero3_nvme.json /workspace/
ENTRYPOINT ["/bin/bash"]