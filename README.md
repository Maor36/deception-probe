# DeceptionProbe: Detecting Lies in LLM Hidden States

**Can we build a lie detector for Large Language Models by reading their internal representations?**

This project investigates whether linear probes trained on hidden state activations can distinguish when an LLM is **lying** (knows the truth but says otherwise), **telling the truth**, or **hallucinating** (doesn't know and makes something up).

## Key Findings

| Stage | Task | Accuracy | Confound-Free | Key Insight |
|-------|------|----------|:---:|-------------|
| 4 | Sycophantic lies (same prompt) | **82.5%** | Yes | Genuine deception signal exists in hidden states |
| 5 | Real-world deception (18 domains) | **70.4%** bal. acc. | Yes | Signal generalizes across domains |
| 6 | Lie vs Hallucination | **100%** | Yes | Lies and hallucinations are completely separable |
| 6 | 3-way (Truth/Lie/Hallucination) | **82.3%** bal. acc. | Yes | Middle layers (16-20) encode deception |

### The Breakthrough Result

> When an LLM **lies** (knows the correct answer but says something else due to sycophantic pressure), its internal state is **completely separable** from when it **hallucinates** (genuinely doesn't know). A simple logistic regression achieves **100% accuracy** distinguishing the two (p=0.0000, 500 permutations).

## Why This Matters

Most existing work on LLM deception detection suffers from a **prompt confound**: the model is explicitly told to lie via a system prompt, so the probe may simply detect the instruction rather than the deception itself. We address this by using **sycophancy** — a natural form of deception where the model spontaneously changes its answer to agree with the user, without any instruction to lie.

### Comparison with Prior Work

| Approach | Method | Our Contribution |
|----------|--------|------------------|
| Azaria & Mitchell (2023) | Instructed deception, single layer | Confound-free design, multi-layer analysis |
| Burns et al. (2023) | CCS on truth directions | Spontaneous (not instructed) deception |
| Pacchiardi et al. (2023) | Behavioral lie detection | Internal state probing, not output-based |
| **This work** | **Spontaneous sycophantic lies** | **Confound-free, lie vs hallucination distinction** |

## Experimental Pipeline

The project is organized into 7 progressive stages:

### Stages 1-3: Establishing Baselines (Confounded)

These stages replicate prior work and **deliberately identify the prompt confound**.

- **Stage 1** — Basic correlation: probe on sycophancy data with different prompts for truth/lie. Result: 100% accuracy (confounded).
- **Stage 2** — Cross-model: same test on Mistral-7B. Result: 100% (still confounded).
- **Stage 3** — Confound analysis: confirms that prompt differences explain the 100% accuracy.

### Stage 4: The Real Test (Confound-Free)

**Same sycophantic prompt** for both conditions. The only difference is what the model *chose* to do:
- **Lie**: model caved to pressure and agreed with the wrong answer
- **Truth**: model resisted pressure and gave the correct answer

**Result: 82.5% accuracy** (chance = 50%, p < 0.001). Best layer: 15.

### Stage 5: Real-World Generalization

Tests deception detection across **18 real-world domains** (medical, legal, financial, etc.) using 459 scenarios where the model must choose between honest and deceptive responses.

**Result: 70.4% balanced accuracy** (chance = 50%, p < 0.001). Best layer: 17.

### Stage 6: Lie vs Hallucination (The Key Experiment)

Three-way classification: **Truth vs Lie vs Hallucination**.

- Questions the model answers correctly = **Truth**
- Questions the model answers correctly but changes under pressure = **Lie**
- Questions the model gets wrong even without pressure = **Hallucination**

**Results:**

| Comparison | Accuracy | Interpretation |
|-----------|----------|----------------|
| 3-way (T/L/H) | 82.3% | Strong overall separation |
| Truth vs Lie | 100% | Model "knows" when it's lying |
| **Lie vs Hallucination** | **100%** | **Completely different internal states** |
| Truth vs Hallucination | 67.4% | Hardest — no internal "tension" in hallucination |
| Length baseline | 60.3% | Signal is not based on response length |

### Stage 7: Advanced Hallucination Detection

Six methods to improve Truth vs Hallucination detection using the hidden states saved from Stage 6:
1. Multi-layer fusion
2. Layer difference vectors
3. Statistical features (norms, variance, entropy)
4. Combined features with PCA
5. Hallucination direction (single vector)
6. Permutation validation

### Layer Profile

Across all confound-free stages, **middle layers (15-20)** consistently outperform early and late layers:

```
Layer  0: 33.3% (embedding — chance level)
Layer  2: 66.6%
Layer  4: 67.6%
...
Layer 16: 81.9%
Layer 17: 82.1%
Layer 18: 82.3%
Layer 20: 82.3% <-- BEST
...
Layer 31: 80.7%
```

Layer 0 at chance confirms the signal is **semantic**, not lexical.

## Quick Start (Google Colab)

### Prerequisites

- Google Colab with **A100 GPU** (or H100)
- HuggingFace account with access to [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Run

```python
# 1. Install dependencies
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn

# 2. Clone repo
!git clone https://github.com/Maor36/deception-probe.git
%cd deception-probe

# 3. Set HuggingFace token
import os
os.environ["HF_TOKEN"] = "your_token_here"

# 4. Run any stage
%run stages/stage4_same_prompt_test/run_stage4.py          # ~30 min, confound-free
%run stages/stage6_hallucination/run_stage6.py             # ~25 min, lie vs hallucination
%run stages/stage7_hallucination_detection/run_stage7.py   # ~5 min, no GPU needed (uses saved data)
```

### Getting a HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read access)
3. Accept the Llama license at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Repository Structure

```
deception-probe/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── stages/                      # All experiment stages
│   ├── stage1_basic_correlation/    # Baseline: sycophancy detection (confounded)
│   ├── stage2_cross_model/          # Cross-architecture validation
│   ├── stage3_accuracy_confounds/   # Confound identification
│   ├── stage4_same_prompt_test/     # Confound-free sycophancy test
│   ├── stage5_realworld_deception/  # 18-domain real-world deception
│   ├── stage6_hallucination/        # Lie vs Hallucination (3-way)
│   └── stage7_hallucination_detection/  # Advanced hallucination methods
├── results/                     # Experiment results and findings
│   ├── FINDINGS.md              # Summary of all results
│   └── stage4_all_layers.txt    # Layer-by-layer Stage 4 results
├── data/                        # Dataset configs (downloaded at runtime)
├── phase1_archive/              # Phase 1 scripts (instructed deception)
└── research_notes/              # Literature review and experiment notes
```

## Method

- **Model**: Llama-3.1-8B-Instruct (4-bit quantized via bitsandbytes)
- **Probe**: Logistic Regression on hidden state activations at the first generated token position
- **Validation**: 5-fold stratified cross-validation with balanced accuracy
- **Statistical tests**: Permutation tests (500 iterations), length baselines
- **Dataset**: [meg-tong/sycophancy-eval](https://huggingface.co/datasets/meg-tong/sycophancy-eval) — 1,817 TriviaQA question pairs

## Confound Controls

Every confound-free stage (4-6) includes:

1. **Same prompt format** for both conditions (no instruction to lie)
2. **Length-only baseline** (consistently near chance: 50-60%)
3. **Permutation tests** (500 iterations, all p < 0.001)
4. **Balanced accuracy** (handles class imbalance)
5. **Embedding layer at chance** (rules out lexical confounds)

## References

- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying*. EMNLP Findings. [arXiv:2304.13734](https://arxiv.org/abs/2304.13734)
- Burns, C. et al. (2023). *Discovering Latent Knowledge in Language Models Without Supervision*. ICLR.
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances*. Computational Linguistics.
- Zou, A. et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency*. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)

## License

Research project — not yet licensed for production use.
