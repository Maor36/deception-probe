# Run Guide: Step-by-Step Execution

This guide explains how to reproduce all results from scratch in Google Colab.

## Prerequisites

1. Google Colab account (Pro recommended for A100 GPU)
2. HuggingFace account with Llama access: [Request access here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. HuggingFace token: [Generate here](https://huggingface.co/settings/tokens)

## Initial Setup (Run Once)

```python
# In a new Colab notebook with GPU runtime:
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn scipy

!git clone https://github.com/Maor36/deception-probe.git
%cd deception-probe

import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
```

## Execution Order

### Phase 1: Baselines (Stages 1-3)

These stages demonstrate the **prompt confound** problem. They achieve 100% accuracy but this is misleading — the probe detects the different prompt format, not deception itself.

```python
%run stages/stage1_basic_correlation/run_stage1.py      # ~15 min
%run stages/stage2_cross_model/run_stage2.py             # ~15 min
%run stages/stage3_accuracy_confounds/run_stage3.py      # ~15 min
```

**Expected:** 100% accuracy on all stages. This is the confound we need to eliminate.

### Phase 2: Confound-Free Detection (Stage 4)

The critical experiment: **same prompt** for truth and lie conditions. The only difference is whether the model chose to lie or resist sycophantic pressure.

```python
%run stages/stage4_same_prompt_test/run_stage4.py        # ~25 min
```

**Expected:** ~82% accuracy (chance = 50%, p < 0.001). This proves a genuine deception signal exists.

### Phase 3: Generalization (Stage 5)

Tests whether the signal generalizes beyond trivia to 18 real-world domains.

```python
%run stages/stage5_realworld_deception/run_stage5_part_a.py   # ~20 min (data generation)
%run stages/stage5_realworld_deception/run_stage5_part_b.py   # ~10 min (analysis)
```

**Expected:** ~70% balanced accuracy across domains.

### Phase 4: Lie vs Hallucination (Stage 6) — Key Experiment

The most novel finding: can we distinguish lies from hallucinations?

```python
%run stages/stage6_hallucination/run_stage6.py           # ~30 min
```

**Expected:** 100% accuracy for Lie vs Hallucination, 82% for 3-way classification.

### Phase 5: Advanced Hallucination Methods (Stage 7)

Six methods to improve Truth vs Hallucination detection. **No GPU needed** — uses saved data from Stage 6.

```python
%run stages/stage7_hallucination_detection/run_stage7.py  # ~5 min, CPU only
```

**Expected:** Best method reaches ~78% (up from 67% baseline).

### Phase 6: Cross-Model Universality (Stage 8) — Key Experiment

Tests the deception signal across three different model architectures. **Important:** This requires running 3 models sequentially. The script has checkpoint/resume — if it crashes, just re-run.

```python
# IMPORTANT: Start with a fresh runtime for this stage
# Runtime → Disconnect and delete runtime → Reconnect

!pip install -q transformers accelerate bitsandbytes datasets scikit-learn scipy
%cd /content/deception-probe
!git pull
import os; os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"

%run stages/stage8_cross_model/run_stage8.py             # ~60 min
```

**Expected:**
- All 3 models: 100% within-model detection
- Llama ↔ Mistral: 98-100% transfer
- Qwen: inverted polarity (2% direct, 97-98% flipped)
- All controls pass (Layer 0 ~50%, Length ~50%)

### Phase 7: Types of Deception (Stage 9)

Tests whether different kinds of lies share the same internal representation.

```python
# Fresh runtime recommended
%run stages/stage9_deception_types/run_stage9.py          # ~40 min
```

**Expected:** Results pending first execution.

### Phase 8: Scale Test (Stage 10) — Optional

Requires A100 80GB GPU. Tests whether the signal exists in Llama-70B.

```python
%run stages/stage10_scale_70b/run_stage10.py              # ~60 min, A100 required
```

## GPU Memory Management

If you encounter GPU out-of-memory errors:

1. **Runtime → Disconnect and delete runtime** (not just restart!)
2. **Runtime → Manage sessions → Delete all sessions**
3. Close all other Colab notebooks
4. Reconnect and verify:
   ```python
   import torch
   free_gb = torch.cuda.mem_get_info()[0] / 1e9
   print(f"Free GPU memory: {free_gb:.1f} GB")
   # Should show ~79 GB for H100 or ~39 GB for A100
   ```

## Verifying Results

After running a stage, check results:

```python
import json
with open("results/stage8_results.json") as f:
    r = json.load(f)
print(json.dumps(r, indent=2))
```

All results are also summarized in `results/FINDINGS.md`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OutOfMemoryError` | Disconnect and delete runtime, close other notebooks |
| `gated repo` error | Accept Llama license on HuggingFace, check HF_TOKEN |
| Stage 7 fails | Run Stage 6 first (Stage 7 needs its saved data) |
| Stage 8 crashes mid-run | Just re-run — checkpoints will skip completed models |
| `ModuleNotFoundError` | Run `!pip install -q transformers accelerate bitsandbytes datasets scikit-learn scipy` |
