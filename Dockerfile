# syntax=docker/dockerfile:1
############################################################
#  Dockerfile for production Pythia training on H100 GPUs  #
#  CUDA 12.4 + flash‑attn 3 + DeepSpeed 0.14 + FP8 Ready     #
############################################################

FROM nvcr.io/nvidia/pytorch:24.06-py3

# ── Basic environment flags ───────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── OS‑level dependencies ────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential \
        python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Python & CUDA stack ───────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# ── LLM / training libraries ─────────────────────────────
RUN pip install transformers==0.22.0 \
                 datasets==2.21.0 \
                 deepspeed==0.17.2 \
                 accelerate==0.29.2 \
                 evaluate wandb tiktoken \
                 xformers==0.0.33 \
                 bitsandbytes==0.43.1

# ── flash‑attn 3 build (SM90/H100) ─────────────────────────────
RUN git clone https://github.com/Dao-AILab/flash-attention.git /opt/flash-attn && \
    cd /opt/flash-attn && \
    pip install ninja && \
    python setup.py install

# ── (Optional) Multi‑GPU: OpenMPI ─────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Create default workspace ─────────────────────────────
WORKDIR /workspace

# Copy default DeepSpeed Stage‑3 config (NVMe offload capable)
# Supply this file next to Dockerfile or add separately.
COPY ds_zero3_nvme.json /workspace/

# ── Entrypoint ────────────────────────────────────────────
ENTRYPOINT ["/bin/bash"]