# DeceptionProbe — Step-by-Step Running Guide

## Prerequisites

1. **Google Colab Pro/Pro+** (for A100 GPU access) or any machine with 16GB+ VRAM
2. **HuggingFace account** with access to Llama-3.1-8B-Instruct
3. **HuggingFace token** (read access)

## Option A: Run from GitHub (Recommended)

### 1. Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com/) and create a new notebook.

### 2. Set GPU Runtime

- Click **Runtime → Change runtime type**
- Select **A100 GPU** (or T4 if A100 is unavailable)
- Click **Save**
- Click **Connect**

### 3. Install Dependencies

```python
# Cell 1: Install packages
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn
```

### 4. Clone Repository

```python
# Cell 2: Clone and set token
!git clone https://github.com/Maor36/deception-probe.git
%cd deception-probe

import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"  # Replace with your token
```

### 5. Run Stage 1

```python
# Cell 3: Run Stage 1 (~20 min on A100)
%run stages/stage1_basic_correlation/run_stage1.py
```

**Expected output:**
- Dataset loading (1,817 questions parsed)
- Model loading (Llama-8B in 4-bit, ~5GB VRAM)
- Generation progress (250 questions x 2 conditions)
- Sycophancy rate (expect 10-30%)
- Probe accuracy per layer (8, 12, 16, 24)
- Confound analysis (length baseline)
- Honest assessment

### 6. Run Stage 2 (if Stage 1 shows signal)

```python
# Cell 4: Run Stage 2 (~15 min per model)
%run stages/stage2_cross_model/run_stage2.py
```

### 7. Run Stage 3 (detailed accuracy)

```python
# Cell 5: Run Stage 3 (~40 min)
%run stages/stage3_accuracy_confounds/run_stage3.py
```

### 8. Run Stage 4 (hallucination vs lie)

```python
# Cell 6: Run Stage 4 (~50 min)
%run stages/stage4_hallucination/run_stage4.py
```

---

## Option B: Run Directly (Without Cloning)

If the repo is private, download scripts directly:

```python
# Cell 1: Install
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn

# Cell 2: Download script
!wget -q "SCRIPT_URL" -O experiment.py && cat experiment.py | head -5

# Cell 3: Run
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
!python experiment.py
```

---

## Interpreting Results

### Stage 1 Output Guide

**Sycophancy Rate:**
- 0-5%: Model is very robust — hard to get enough lie samples
- 5-15%: Normal range for Llama-8B
- 15-30%: Good amount of data for training probes
- 30%+: Model is very sycophantic

**Probe Accuracy:**
- 50%: Chance level — no signal (probe cannot detect lies)
- 55-65%: Weak signal — similar to published results
- 65-80%: Moderate signal — promising
- 80%+: Strong signal — if confounds are clean, this is significant

**Confound Checks:**
- Length baseline < 60%: Good — length doesn't explain the signal
- Length baseline > 70%: Bad — the probe might just be detecting short vs long answers
- p-value < 0.05: Statistically significant
- p-value > 0.05: Could be random noise

### Decision Tree

```
Stage 1 accuracy > 70% AND p < 0.05 AND length baseline < 65%?
├── YES → Run Stage 2 (cross-model)
│   ├── Works on 2+ models → Run Stage 3 (confounds)
│   │   ├── Passes confounds → Run Stage 4 (hallucination)
│   │   └── Fails confounds → Report honestly, try different approach
│   └── Only works on 1 model → Model-specific artifact
└── NO → Try different layers, more data, or different pressure method
```

---

## Troubleshooting

### "Model requires authentication"
Set your HuggingFace token:
```python
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
```

### "CUDA out of memory"
The model uses 4-bit quantization (~5GB). If still OOM:
- Reduce `MAX_QUESTIONS` in the script
- Reduce `MAX_NEW_TOKENS`
- Use T4 instead of A100 (still works, just slower)

### "0 sycophantic samples"
The model is too robust to sycophancy pressure. Options:
- Increase `MAX_QUESTIONS` (more chances to catch lies)
- Try a different model (some are more sycophantic)
- Use stronger pressure prompts

### "Parsed 0 valid question pairs"
Dataset format mismatch. The scripts are designed for meg-tong/sycophancy-eval format. If using a different dataset, adjust the parsing code.

---

## Dataset Format Reference

### meg-tong/sycophancy-eval (answer.jsonl)

Each row has 3 fields:
- `prompt`: List of dicts `[{"type": "human", "content": "..."}]`
- `base`: Dict with `{"dataset", "question", "answer", "correct_answer", "incorrect_answer"}`
- `metadata`: Dict with template info

Every 4 consecutive rows = 1 question:
- Row 0: Neutral (just the question)
- Row 1: "I don't think [correct answer]" (pressure against correct)
- Row 2: "I think [incorrect answer]" (pressure toward wrong) ← **This is the sycophancy condition**
- Row 3: "I think [correct answer]" (control — pressure toward correct)

### HaluEval (halueval_qa.jsonl)

Each row has:
- `knowledge`: Background knowledge text
- `question`: The question
- `right_answer`: Correct answer
- `hallucinated_answer`: Plausible but wrong answer
