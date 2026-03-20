# Data Locations

This document maps every data artifact in the project to its storage location.
Large binary files (hidden states) are stored in Google Drive; metadata and labels are in this Git repository.

## Git Repository (this repo)

### Experiment 02 — Confound-Free Detection

| File | Description |
|------|-------------|
| `results/exp02c_responses.json` | 915 scenario responses with Phase A/B outputs, keyword labels, metadata |
| `results/exp02c_token_labels.json` | Token-level deception labels (0=neutral, 1=deceptive) — LLM judge: 590 lied, 325 resisted |
| `results/exp02d_token_probe.json` | Token-level probe metrics per layer (balanced accuracy, precision, recall, F1) |
| `results/exp02d_streaming_sim.json` | Streaming polygraph simulation — 5 aggregation methods × 4 window sizes |
| `results/exp02d_cross_phase.json` | Cross-phase transfer results (trivia → real-world) per layer and strategy |
| `results/exp02d_probe_weights.npz` | Trained probe weights for production deployment (Layer 12, 4096 dims) |
| `results/exp02a_*.json` | Step 2A trivia sycophancy results |

### Experiment 04 — Cross-Model Transfer

| File | Description |
|------|-------------|
| `results/exp04_cross_model.json` | Cross-model transfer results: within-model probes + cross-model transfer for 3 models × 2 token strategies |
| `results/exp04_checkpoint_v3.pkl` | Checkpoint with raw hidden states for all 3 models (for re-analysis without re-running) |

**Token strategies in exp04:**
- `first_gen` / `last_prompt` — Both extract the **last prompt token** hidden state (same vector). This is the standard approach used by Burns et al. (CCS) and Li et al. (ITI). The two names refer to the same position because in HuggingFace `generate()`, the last position of the prefill step produces the first generated token's logits.
- `mean_all` — Mean of all generated token hidden states. Standard approach used by Azaria & Mitchell and Goldowsky-Dill et al.

### Scripts

| Directory | Description |
|-----------|-------------|
| `experiments/02_confound_free_detection/` | Steps 2a–2d scripts |
| `experiments/04_cross_model_transfer/` | Cross-model transfer script (v3) |
| `colab_setup.py` | Pre-flight check for Colab sessions |

## Google Drive

**Base path:** `/content/drive/MyDrive/deception-probe-results/`

| File / Folder | Size | Description |
|---------------|------|-------------|
| `exp02c_token_hs/` | ~2–3 GB | 915 `.npz` files, one per sample. Each contains hidden states for **every generated token** across layers 12, 14, 15, 16, 18. Key format: `layer_{n}` → numpy array of shape `(n_tokens, 4096)` |
| `exp02c_sentence_hs.npz` | ~50 MB | Sentence-level hidden states (divergence token only) for all 915 samples |
| `exp02c_responses.json` | ~5 MB | Copy of responses JSON (backup) |
| `exp02c_token_labels.json` | ~3 MB | Copy of token labels JSON (backup) |
| `exp02d_*.json` / `exp02d_*.npz` | small | Copies of step2d results |
| `exp04_cross_model.json` | small | Copy of exp04 results |
| `exp04_checkpoint_v3.pkl` | ~50 MB | Copy of exp04 checkpoint (raw hidden states) |

### How to access in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
hs_dir = "/content/drive/MyDrive/deception-probe-results/exp02c_token_hs"
sample = np.load(f"{hs_dir}/sample_0000.npz")
print(sample.files)          # ['layer_12', 'layer_14', 'layer_15', 'layer_16', 'layer_18']
print(sample['layer_15'].shape)  # (n_tokens, 4096)
```

### Save exp04 results to Drive (run in Colab after exp04)

```python
!cp /content/deception-probe/results/exp04_*.json /content/drive/MyDrive/deception-probe-results/
!cp /content/deception-probe/results/exp04_*.pkl /content/drive/MyDrive/deception-probe-results/
```

### Push exp04 results to Git (run in Colab after exp04)

```python
!cd /content/deception-probe && git add -f results/exp04_* && git commit -m "results: exp04 cross-model transfer" && git push
```

### Save step2d results to Drive (run in Colab after step2d)

```python
!cp /content/deception-probe/results/exp02d_*.json /content/drive/MyDrive/deception-probe-results/
!cp /content/deception-probe/results/exp02d_*.npz /content/drive/MyDrive/deception-probe-results/
```

### Push step2d results to Git (run in Colab after step2d)

```python
!cd /content/deception-probe && git add -f results/exp02d_* && git commit -m "results: exp02d analysis" && git push
```

## Notes

- Hidden states are too large for Git (~2–3 GB total). Always keep the Google Drive backup.
- If you re-run `step2c_collect_realworld.py`, new hidden states will overwrite the Drive folder.
- `step2d_analyze_realworld.py` expects hidden states in `results/exp02c_token_hs/`. Use symlinks from Drive if running in Colab.
- The LLM judge (step2c_label) updated labels from 250/665 to 590/325. The `sentence_hs.npz` still has old labels (250/665) — the token_labels.json has the correct ones.
- In exp04, `first_gen` and `last_prompt` are identical (same hidden state vector). This is expected — see token strategies note above.
