# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="90" \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=0 

# ── OS & Build tools ─────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl build-essential python3-pip cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# ── Core LLM tools ───────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      triton==3.2.0 \
      transformers==4.52.4 \
      datasets==2.21.0 \
      deepspeed==0.17.2 \
      accelerate==0.29.2 \
      evaluate wandb tiktoken \
      bitsandbytes==0.43.3 \
      packaging ninja

# ── Flash‑Attention‑3 (source build) ─────────────────────
RUN git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn && \
    cd /tmp/flash-attn/hopper && \
    python setup.py install && \
    cd / && rm -rf /tmp/flash-attn

# ── Unsloth 最新版 (FA3 対応) ─────────────────────────────
RUN pip install --no-cache-dir \
      "unsloth[cu124-ampere-torch240]" \
      unsloth-zoo

WORKDIR /workspace
COPY ds_zero3_nvme.json /workspace/
ENTRYPOINT ["/bin/bash"]