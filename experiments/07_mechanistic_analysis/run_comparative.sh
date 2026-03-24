#!/bin/bash
# ============================================================================
# Experiment 07: Comparative Mechanistic Analysis
# ============================================================================
# This script runs both mechanistic experiments in sequence:
#   07a: Comparative Logit Lens (cheaper, ~45-60 min)
#   07b: Comparative Activation Patching (expensive, ~90-120 min)
#
# REQUIREMENTS:
#   - GPU with at least 16GB VRAM (4-bit quantized) or 32GB (float16)
#   - HuggingFace token with access to Llama-3.1-8B-Instruct
#   - Python packages: see requirements.txt + matplotlib
#
# USAGE (must be run from repo root, or use this script directly):
#   export HF_TOKEN="your_huggingface_token"
#   bash experiments/07_mechanistic_analysis/run_comparative.sh
#
# RECOMMENDED PLATFORMS:
#   - Google Colab Pro (A100 40GB) — ~$10/month
#   - Lambda Labs (A100 80GB) — ~$1.10/hour
#   - Vast.ai (A100/A6000) — ~$0.50-1.00/hour
#   - RunPod (A100) — ~$0.80/hour
# ============================================================================

set -e

# Navigate to repo root FIRST (before any file operations)
cd "$(dirname "$0")/../.."
REPO_ROOT=$(pwd)
echo "Working directory: $REPO_ROOT"

echo "============================================"
echo "Experiment 07: Comparative Mechanistic Analysis"
echo "============================================"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. You may need it for Llama access."
    echo "  Set it with: export HF_TOKEN='your_token'"
    echo ""
fi

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\"}')" 2>/dev/null || echo "GPU check failed"
python3 -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB') if torch.cuda.is_available() else print('No CUDA')" 2>/dev/null || true
echo ""

# Check matplotlib is installed (required for plotting)
python3 -c "import matplotlib" 2>/dev/null || {
    echo "Installing matplotlib (required for plots)..."
    pip install matplotlib
}

# Create results directories (AFTER cd to repo root)
mkdir -p results/figures

# Run 07a: Comparative Logit Lens
echo "============================================"
echo "Running 07a: Comparative Logit Lens..."
echo "  (Expected runtime: 45-60 minutes)"
echo "============================================"
python3 experiments/07_mechanistic_analysis/comparative_logit_lens.py
echo ""
echo "07a complete! Check results/exp07a_comparative_logit_lens.json"
echo ""

# Run 07b: Comparative Activation Patching
echo "============================================"
echo "Running 07b: Comparative Activation Patching..."
echo "  (Expected runtime: 90-120 minutes)"
echo "============================================"
python3 experiments/07_mechanistic_analysis/comparative_activation_patching.py
echo ""
echo "07b complete! Check results/exp07b_comparative_activation_patching.json"
echo ""

echo "============================================"
echo "All experiments complete!"
echo ""
echo "Results:"
echo "  results/exp07a_comparative_logit_lens.json"
echo "  results/exp07b_comparative_activation_patching.json"
echo "  results/figures/exp07a_flip_layer_comparison.png"
echo "  results/figures/exp07a_rank_trajectories.png"
echo "  results/figures/exp07b_recovery_comparison.png"
echo "============================================"
