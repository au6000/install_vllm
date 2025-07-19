# syntax=docker/dockerfile:1
############################################################
#  Dockerfile – Unsloth + Flash‑Attn‑3 on NVIDIA H100      #
#  CUDA 12.4  •  PyTorch 2.4.0 • Triton 3.2  •  DS 0.17    #
############################################################
FROM nvcr.io/nvidia/pytorch:24.06-py3

# ---------- Basic environment -------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="90"        \
    # Flash‑Attn 3 needs AMP “fp8e5m2“ support on sm90
    FLASH_ATTENTION_SKIP_CUDA_BUILD=0

# ---------- OS packages --------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential \
        python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ---------- PyTorch stack -----------------------------------------------
RUN pip install -U pip setuptools wheel && \
    pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# ---------- Core libs ----------------------------------------------------
RUN pip install --no-cache-dir \
        triton==3.2.0          \
        deepspeed==0.17.2      \
        accelerate==0.29.2     \
        bitsandbytes==0.43.1   \
        transformers==0.22.0   \
        datasets==2.21.0       \
        evaluate wandb tiktoken xformers==0.0.33

# ---------- Flash‑Attention 3 -------------------------------------------
# 3.0.0.post1 wheels exist for sm90; if unavailable, fallback to build.
RUN pip install --no-cache-dir "flash-attn>=3.0.0" || true
#  └─ If wheel missing, compile from source
RUN test -e "$(python - <<'PY' ; import importlib, sys, pkg_resources as pr ; \
        sys.exit(0 if importlib.util.find_spec('flash_attn') else 1) ; PY)" || \
    (git clone https://github.com/Dao-AILab/flash-attention.git /opt/flash-attn && \
     cd /opt/flash-attn && git checkout v3.0.0 && \
     python setup.py install)

# ---------- Unsloth (Flash‑Attn 3 対応版) ---------------------------------
RUN pip install --no-cache-dir "unsloth[gpu]"==0.6.0

# ---------- Smoke‑test kernel loading -----------------------------------
RUN python - <<'PY'
import torch, unsloth, flash_attn_cuda
print("CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
print("Triton:", torch.version.triton, "Flash‑Attn:", flash_attn_cuda.__version__)
print("Unsloth compiled:", unsloth.utils.get_compiled())
PY

# ---------- (Optional) MPI for multi‑node --------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY ds_zero3_nvme.json /workspace/
ENTRYPOINT ["/bin/bash"]