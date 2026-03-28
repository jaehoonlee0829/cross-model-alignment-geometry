#!/bin/bash
# Setup script for GPU cloud environment
# Run this after SSH-ing into your GPU instance
set -e

echo "=== Cross-Model Activation Oracles — GPU Environment Setup ==="

# 1. Clone repo (replace with your actual repo URL)
# git clone https://github.com/YOUR_USERNAME/cross-model-activation-oracles.git
# cd cross-model-activation-oracles

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (CUDA 12.1 — adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Login to HuggingFace (for gated models like Gemma)
echo ""
echo "You'll need to login to HuggingFace for Gemma access:"
echo "  huggingface-cli login"
echo ""
echo "Make sure you've accepted the Gemma license at:"
echo "  https://huggingface.co/google/gemma-2-2b"

# 6. Login to Weights & Biases (optional)
# wandb login

# 7. Verify GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 8. Quick sanity check — can we load the models?
python3 -c "
from transformers import AutoConfig
for model in ['google/gemma-2-2b', 'Qwen/Qwen2.5-1.5B']:
    try:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        print(f'{model}: d_model={cfg.hidden_size}, n_layers={cfg.num_hidden_layers}')
    except Exception as e:
        print(f'{model}: FAILED — {e}')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run the pipeline:"
echo "  1. python scripts/run_extraction.py --config configs/default.yaml"
echo "  2. python scripts/run_cka.py --config configs/default.yaml"
echo "  3. python scripts/run_alignment.py --config configs/default.yaml --alignment-only"
echo "  4. python scripts/run_alignment.py --config configs/default.yaml  # (with oracle)"
