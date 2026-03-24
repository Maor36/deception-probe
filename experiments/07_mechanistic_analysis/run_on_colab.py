"""
Quick-start script for running Experiment 07 on Google Colab.

Copy-paste the cells below into a Colab notebook.

INSTRUCTIONS:
    1. Open Google Colab (colab.research.google.com)
    2. Select Runtime → Change runtime type → T4 GPU (free) or A100 (Pro)
    3. Run the cells below in order

ESTIMATED TIME:
    - Colab Free (T4 16GB): ~2-3 hours total (4-bit quantized)
    - Colab Pro (A100 40GB): ~1.5-2 hours total (float16)
    - Colab Pro+ (A100 80GB): ~1-1.5 hours total (float16, best results)
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────
# !pip install transformers accelerate bitsandbytes datasets scikit-learn torch scipy tqdm matplotlib

# ── Cell 2: Clone repo ───────────────────────────────────────────────────
# !git clone https://github.com/cohen-liel/deception-probe.git
# %cd deception-probe

# ── Cell 3: Set HuggingFace token ────────────────────────────────────────
# import os
# os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"  # Get from huggingface.co/settings/tokens

# ── Cell 4: Check GPU ────────────────────────────────────────────────────
# import torch
# if torch.cuda.is_available():
#     gpu = torch.cuda.get_device_name(0)
#     vram = torch.cuda.get_device_properties(0).total_mem / 1e9
#     print(f"GPU: {gpu} ({vram:.1f} GB)")
#     if vram >= 32:
#         print("→ Excellent! Can run in float16 (best quality)")
#     elif vram >= 16:
#         print("→ Good! Will use 4-bit quantization")
#     else:
#         print("→ Warning: May need smaller batch size")
# else:
#     print("NO GPU! Go to Runtime → Change runtime type → GPU")

# ── Cell 5: Run Experiment 07a (Comparative Logit Lens) ──────────────────
# %run experiments/07_mechanistic_analysis/comparative_logit_lens.py

# ── Cell 6: Run Experiment 07b (Comparative Activation Patching) ─────────
# %run experiments/07_mechanistic_analysis/comparative_activation_patching.py

# ── Cell 7: View results ─────────────────────────────────────────────────
# import json
# from IPython.display import Image, display
#
# # --- Logit Lens results ---
# with open("results/exp07a_comparative_logit_lens.json") as f:
#     r07a = json.load(f)
# print("=" * 60)
# print("LOGIT LENS: Where does each lie type originate?")
# print("=" * 60)
# for dtype, data in r07a["deception_types"].items():
#     median = data.get("median_flip_layer")
#     n = data.get("n_lie_examples", 0)
#     if median is not None:
#         print(f"  {dtype}: median flip at layer {median:.0f} ({n} examples)")
#     else:
#         print(f"  {dtype}: no lie examples found")
# stat = r07a.get("statistical_comparison", {})
# kw = stat.get("kruskal_wallis")
# if kw and kw.get("p") is not None:
#     print(f"\nKruskal-Wallis: H={kw['H']:.2f}, p={kw['p']:.4f}")
#     print("*** SIGNIFICANT ***" if kw['p'] < 0.05 else "Not significant")
#
# # --- Activation Patching results ---
# with open("results/exp07b_comparative_activation_patching.json") as f:
#     r07b = json.load(f)
# print("\n" + "=" * 60)
# print("ACTIVATION PATCHING: Causal layer per lie type")
# print("=" * 60)
# for dtype, data in r07b["deception_types"].items():
#     bl = data.get("best_layer")
#     br = data.get("best_recovery_rate")
#     n = data.get("n_lie_examples", 0)
#     if bl is not None and br is not None:
#         print(f"  {dtype}: most causal layer = {bl} ({br*100:.0f}% recovery, {n} examples)")
#     else:
#         print(f"  {dtype}: no data (model may not have lied for this type)")
# comp = r07b.get("comparison", {})
# spread = comp.get("layer_spread")
# if spread is not None:
#     print(f"\nLayer spread: {spread} layers")
#     if spread >= 3:
#         print("*** STRONG EVIDENCE: Different deception types use DIFFERENT circuits! ***")
#     elif spread >= 1:
#         print("** MODERATE: Some circuit differentiation detected **")
#     else:
#         print("Deception types appear to use similar circuits")
#
# # --- Show plots ---
# import os
# for fig_path in [
#     "results/figures/exp07a_flip_layer_comparison.png",
#     "results/figures/exp07a_rank_trajectories.png",
#     "results/figures/exp07b_recovery_comparison.png",
# ]:
#     if os.path.exists(fig_path):
#         print(f"\n{fig_path}:")
#         display(Image(fig_path))
#     else:
#         print(f"\n{fig_path}: not found (experiment may not have completed)")

if __name__ == "__main__":
    print("This file contains Colab instructions.")
    print("Copy the commented cells above into a Google Colab notebook.")
    print("")
    print("Or run directly from repo root:")
    print("  python experiments/07_mechanistic_analysis/comparative_logit_lens.py")
    print("  python experiments/07_mechanistic_analysis/comparative_activation_patching.py")
