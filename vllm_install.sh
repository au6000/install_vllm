#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  echo "[`date +%T`] $1"
}

ENV_NAME="vllm"
CONDA_PREFIX="$HOME/miniforge"
CONDA_SH="$CONDA_PREFIX/etc/profile.d/conda.sh"

log "🔧 Loading CUDA & cuDNN modules"
module load cuda/12.4
module load cudnn/9.6.0

log "📦 Checking / installing Miniforge..."
if [ ! -d "$CONDA_PREFIX" ]; then
  log "🚀 Installing Miniforge silently..."
  curl -Ls https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
       -o /tmp/miniforge.sh
  chmod +x /tmp/miniforge.sh
  yes | /tmp/miniforge.sh -b -p "$CONDA_PREFIX"
fi

log "🔧 Initializing conda shell"
set +u
if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge/etc/profile.d/conda.sh"
else
  log "❌ conda.sh not found at $HOME/miniforge/etc/profile.d/conda.sh"
  exit 1
fi
set -u

if ! command -v conda &>/dev/null; then
  log "❌ conda not found after sourcing conda.sh"
  log "PATH is: $PATH"
  exit 1
fi

if ! command -v conda &>/dev/null; then
  log "❌ conda not found after sourcing $CONDA_SH"
  exit 1
fi

log "⚙️  Disabling auto_activate_base"
conda config --set auto_activate_base false

if ! conda env list | grep -q "^$ENV_NAME "; then
  log "📦 Creating environment '$ENV_NAME'"
  conda create -y -n "$ENV_NAME" python=3.10
else
  log "📦 Environment '$ENV_NAME' already exists"
fi

log "🚀 Activating environment '$ENV_NAME'"
conda activate "$ENV_NAME" || {
  log "❌ Failed to activate environment '$ENV_NAME'"
  exit 1
}

log "🐍 Python path: $(which python)"
log "📍 Conda env:  $(conda info --envs | grep '^'"$ENV_NAME")"

log "📦 Installing Python packages"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1 triton==3.2.0
pip install flash-attn==2.7.4.post1

log "🧪 Verifying flash-attn installation"
python - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_interface as fai
print("torch          :", torch.__version__)
print("flash-attn ver :", flash_attn.__version__)
print("flash_attn_func:", fai.flash_attn_func)
PY

log "🧪 Verifying vllm generation"
python - <<'PY'
from vllm import LLM, SamplingParams
prompts = ["Hello, my name is"]
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, SamplingParams(max_tokens=16))
print("Generated text:", outputs[0].outputs[0].text)
PY

log "✅ install.sh completed successfully"