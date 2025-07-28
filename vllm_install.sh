#!/usr/bin/env bash

set -Eeuo pipefail


ENV_NAME="vllm"

PYTHON_VERSION="3.10"

INSTALL_DIR="$HOME/miniconda3"

log() {

  echo "[$(date +%T)] $1"
}


log "🔧 Loading CUDA & cuDNN modules..."
module load cuda/12.4
module load cudnn/9.6.0
log "👍 Modules loaded."

log "📦 Checking for any Conda installation (Miniconda, Anaconda, etc.)..."

if command -v conda &> /dev/null; then
    log "👍 Conda is already installed."
    CONDA_BASE=$(conda info --base)
    log "Found Conda base at: $CONDA_BASE"
else
    log "🚀 Conda not found. Installing Miniconda to '$INSTALL_DIR'..."

    curl -Ls https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -o /tmp/miniconda.sh
    chmod +x /tmp/miniconda.sh


    /bin/bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"
    rm /tmp/miniconda.sh

    log "🔧 Initializing Conda for your shell. This will modify your shell's rc file (e.g., ~/.bashrc)."

    "$INSTALL_DIR/bin/conda" init bash

    CONDA_BASE="$INSTALL_DIR"
    log "👍 Miniconda installed successfully at '$CONDA_BASE'."
fi


log "🔧 Initializing Conda shell functions for this script session..."

set +u
CONDA_SH_PATH="$CONDA_BASE/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then
    source "$CONDA_SH_PATH"
else
    log "❌ ERROR: conda.sh not found at '$CONDA_SH_PATH'."
    log "Your Conda installation might be corrupted or in an unexpected location."
    exit 1
fi

set -u


if ! command -v conda &> /dev/null; then
    log "❌ ERROR: 'conda' command is still not available after sourcing conda.sh."
    log "       This can happen if the 'conda init' command hasn't taken effect."
    log "       Please try running this script again in a new terminal session."
    log "       Current PATH is: $PATH"
    exit 1
fi

-
log "⚙️ Disabling the automatic activation of the 'base' environment."
conda config --set auto_activate_base false

log "🐍 Checking for Conda environment '$ENV_NAME'..."

if ! conda env list | grep -q "^$ENV_NAME "; then
    log "📦 Environment '$ENV_NAME' not found. Creating it with Python $PYTHON_VERSION..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
else
    log "👍 Environment '$ENV_NAME' already exists."
fi

log "✅ Conda setup is complete."

log "🚀 Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME" || {
    log "❌ Failed to activate environment '$ENV_NAME'."
    exit 1
}
log "👍 Environment activated."

log "🐍 Python path: $(which python)"
log "📍 Conda env info: $(conda info --envs | grep --color=never "^$ENV_NAME")"

log "📦 Installing Python packages..."
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1 triton==3.2.0
pip install flash-attn==2.7.4.post1
log "👍 Python packages installed."

log "🧪 Verifying flash-attn installation..."
python - <<'PY'
try:
    import torch, flash_attn
    from flash_attn import flash_attn_interface as fai
    print("  - torch version      :", torch.__version__)
    print("  - flash-attn version :", flash_attn.__version__)
    print("  - flash_attn_func is available.")
    print("✅ flash-attn verification PASSED.")
except ImportError as e:
    print(f"❌ flash-attn verification FAILED: {e}")
    exit(1)
PY

log "🧪 Verifying vLLM generation..."
python - <<'PY'
try:
    from vllm import LLM, SamplingParams
    prompts = ["Hello, my name is"]
    # 小さなモデルでテスト実行
    llm = LLM(model="facebook/opt-125m", tensor_parallel_size=1)
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=16)
    outputs = llm.generate(prompts, params)
    generated_text = outputs[0].outputs[0].text
    print(f"  - Prompt: '{prompts[0]}'")
    print(f"  - Generated text: '{generated_text.strip()}'")
    print("✅ vLLM generation verification PASSED.")
except Exception as e:
    print(f"❌ vLLM generation verification FAILED: {e}")
    exit(1)
PY

log "🎉✨ Script completed successfully! ✨🎉"
