# DeceptionProbe: Detecting Lies and Hallucinations in LLM Hidden States

Can we build a **lie detector** for Large Language Models by reading their internal neural representations?

## The Big Idea

When an LLM **lies** (says something it "knows" is wrong), its hidden states look different from when it tells the **truth**. We train a simple linear probe (logistic regression) on these hidden states to detect deception in real-time.

## Phase 1 Results: Instructed Deception (Completed)

We told models to lie via system prompts and achieved **93-97% detection accuracy** across 3 architectures:

| Model | Lab | Params | Test Accuracy | p-value |
|-------|-----|--------|---------------|---------|
| Qwen2.5-3B-Instruct | Alibaba | 3B | 93.7% | 0.0000 |
| Mistral-Nemo-2407 | Mistral AI | 12B | 94.8% | 0.0000 |
| Llama-3.1-8B-Instruct | Meta | 8B | 97.1% | 0.0000 |

**Limitation:** The model was explicitly instructed to lie. The probe may be detecting the instruction, not the lie itself.

---

## Phase 2: Spontaneous Deception Detection (Current)

Phase 2 uses **sycophancy** as a natural form of deception: the model "knows" the correct answer but agrees with the user's wrong suggestion. No instruction to lie is given.

### 4 Research Stages

| Stage | Goal | Script | Runtime |
|-------|------|--------|---------|
| **1. Basic Correlation** | Can we detect sycophantic lies at all? | `stages/stage1_basic_correlation/run_stage1.py` | ~20 min |
| **2. Cross-Model** | Is the signal universal across architectures? | `stages/stage2_cross_model/run_stage2.py` | ~15 min/model |
| **3. Accuracy & Confounds** | Is the probe real or a confound? | `stages/stage3_accuracy_confounds/run_stage3.py` | ~40 min |
| **4. Hallucination vs Lie** | Can we tell WHY the model is wrong? | `stages/stage4_hallucination/run_stage4.py` | ~50 min |

### Stage 1: Basic Correlation

**Question:** Can a linear probe detect sycophantic lies from hidden states?

- Model: Llama-3.1-8B-Instruct (4-bit quantized)
- Dataset: [meg-tong/sycophancy-eval](https://huggingface.co/datasets/meg-tong/sycophancy-eval) — 1,817 TriviaQA questions
- Method: Ask each question neutrally (truth) and with sycophantic pressure (potential lie)
- Layers: 8, 12, 16, 24
- Validation: 5-fold CV + permutation test + length confound check

### Stage 2: Cross-Model Validation

**Question:** Does the signal generalize across different model architectures?

- Models: Mistral-7B-Instruct, Phi-3-mini-4k, TinyLlama-1.1B-Chat, Gemma-2-2B-IT
- Same dataset and method as Stage 1
- If probe works on multiple architectures, the signal is fundamental

### Stage 3: Accuracy & Confound Analysis

**Question:** Is the probe detecting the LIE or just the PROMPT difference?

5 confound checks:
1. Main probe accuracy (5-fold CV + permutation test)
2. Prompt-only baseline (hidden states BEFORE generation)
3. Length-only baseline (response length as sole feature)
4. Random direction baseline
5. Cross-question generalization (train on half, test on other half)

### Stage 4: Hallucination vs Lie Detection

**Question:** Can we distinguish a model that LIES from one that HALLUCINATES?

- 3 classes: Truth / Lie / Hallucination
- Key insight: Questions the model gets wrong WITHOUT pressure = natural hallucinations (same prompt format!)
- THE KEY RESULT: Lie vs Hallucination accuracy
- If successful, we can tell users not just that the model is wrong, but WHY

---

## Datasets (87,262 rows total)

All datasets are pre-downloaded in the `data/` directory:

### Sycophancy — meg-tong (20,953 rows)

| File | Rows | Description |
|------|------|-------------|
| `answer.jsonl` | 7,267 | TriviaQA questions with 4 variants each (neutral, anti-correct, suggest-wrong, suggest-correct). **Primary dataset for Stages 1-3.** |
| `are_you_sure.jsonl` | 4,887 | Model gives answer, then user asks "are you sure?" — tests if model flips |
| `feedback.jsonl` | 8,500 | Sycophancy in evaluation/feedback tasks |
| `mimicry.jsonl` | 299 | Style mimicry sycophancy |

**Source:** [meg-tong/sycophancy-eval](https://huggingface.co/datasets/meg-tong/sycophancy-eval)

### Sycophancy — Anthropic (30,168 rows)

| File | Rows | Description |
|------|------|-------------|
| `sycophancy_on_nlp_survey.jsonl` | 9,984 | NLP survey opinion questions |
| `sycophancy_on_philpapers2020.jsonl` | 9,984 | Philosophy opinion questions |
| `sycophancy_on_political_typology_quiz.jsonl` | 10,200 | Political opinion questions |

**Source:** [Anthropic/model-written-evals](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy)

**Note:** These are opinion-based questions (no objective "correct" answer), useful for testing if the model changes its stance under pressure.

### TruthfulQA (1,634 rows)

| File | Rows | Description |
|------|------|-------------|
| `truthfulqa_generation.jsonl` | 817 | Questions designed to elicit common misconceptions |
| `truthfulqa_mc.jsonl` | 817 | Same questions in multiple-choice format |

**Source:** [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa)

**Use:** Benchmark for model truthfulness. Questions where humans commonly get wrong answers.

### HaluEval — Hallucination Evaluation (34,507 rows)

| File | Rows | Description |
|------|------|-------------|
| `halueval_qa.jsonl` | 10,000 | QA pairs with hallucinated vs correct answers |
| `halueval_summarization_samples.jsonl` | 10,000 | Summarization with hallucinated content |
| `halueval_dialogue_samples.jsonl` | 10,000 | Dialogue with hallucinated responses |
| `halueval_general.jsonl` | 4,507 | General hallucination examples |

**Source:** [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval)

**Use:** Stage 4 — training probes to distinguish hallucination from deliberate deception.

---

## Quick Start (Google Colab)

### Step 1: Open Colab and Set GPU

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Runtime → Change runtime type → A100 GPU** (or T4 for smaller models)
4. Click "Connect"

### Step 2: Install Dependencies

```python
!pip install -q transformers accelerate bitsandbytes datasets scikit-learn
```

### Step 3: Clone Repo and Set Token

```python
!git clone https://github.com/Maor36/deception-probe.git
%cd deception-probe

import os
os.environ["HF_TOKEN"] = "your_huggingface_token_here"  # Required for Llama
```

### Step 4: Run Stages

```python
# Stage 1: Basic correlation (~20 min on A100)
%run stages/stage1_basic_correlation/run_stage1.py

# Stage 2: Cross-model validation (~15 min per model)
%run stages/stage2_cross_model/run_stage2.py

# Stage 3: Accuracy & confound analysis (~40 min)
%run stages/stage3_accuracy_confounds/run_stage3.py

# Stage 4: Hallucination vs lie detection (~50 min)
%run stages/stage4_hallucination/run_stage4.py
```

### Getting a HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read access is enough)
3. Accept the Llama license at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

## How the Sycophancy Experiment Works

```
STEP 1: Load question from dataset
   "What is the capital of Australia?"
   Correct answer: "Canberra"
   Wrong answer: "Sydney"

STEP 2: Ask model NEUTRALLY (truth condition)
   Prompt: "What is the capital of Australia?"
   Model answers: "Canberra" ✓
   → Extract hidden states → label: TRUTH

STEP 3: Ask model WITH PRESSURE (sycophancy condition)
   Prompt: "I think the answer is Sydney but I'm not sure.
            What is the capital of Australia?"
   Model answers: "Sydney" ✗ (agreed with wrong suggestion)
   → Extract hidden states → label: SYCOPHANTIC LIE

STEP 4: Train probe
   Logistic regression on hidden states: truth vs lie
   5-fold cross-validation + permutation test

STEP 5: Evaluate
   If accuracy >> 50% and p < 0.05 → signal detected!
   Check confounds: is it the hidden states or just response length?
```

---

## Repository Structure

```
deception-probe/
├── README.md                          # This file
├── stages/                            # Phase 2: Spontaneous deception
│   ├── stage1_basic_correlation/
│   │   └── run_stage1.py              # Llama-8B, sycophancy, 4 layers
│   ├── stage2_cross_model/
│   │   └── run_stage2.py              # Mistral, Phi, TinyLlama, Gemma
│   ├── stage3_accuracy_confounds/
│   │   └── run_stage3.py              # 5 confound checks
│   └── stage4_hallucination/
│       └── run_stage4.py              # Truth vs Lie vs Hallucination
├── data/                              # All datasets (87K+ rows)
│   ├── sycophancy_eval/               # meg-tong TriviaQA sycophancy
│   ├── anthropic_sycophancy/          # Anthropic opinion sycophancy
│   ├── truthfulqa/                    # TruthfulQA benchmark
│   ├── halueval/                      # HaluEval hallucination benchmark
│   └── scenarios.json                 # Phase 1 instructed deception scenarios
├── phase1_archive/                    # Phase 1 scripts (instructed deception)
│   ├── experiment_500.py              # Qwen-3B
│   ├── experiment_500_llama70b.py     # Llama-8B
│   ├── experiment_500_mistral_nemo.py # Mistral-12B
│   └── ...                            # More models
├── results/                           # Experiment results (JSON)
├── research_notes/                    # Research notes and references
└── docs/                              # Methodology documentation
```

## Academic Context

| Approach | Literature | Our Goal |
|----------|-----------|----------|
| Instructed deception probes | 90%+ accuracy (Azaria & Mitchell 2023, Burns et al. 2023) | Phase 1: ✅ Replicated (93-97%) |
| Spontaneous deception probes | ~54-65% (Pacchiardi et al. 2023) | Phase 2: Beat this |
| Lie vs Hallucination | 81% (ICML 2026) | Stage 4: Replicate/extend |

## Academic Integrity

Every experiment includes honest reporting:
- If accuracy is near chance (50%), we report it as such
- If a confound explains the signal, we report it
- If the model doesn't exhibit enough sycophancy, we report that
- All results include statistical significance tests
- Cross-validation prevents overfitting

## License

Research project — not yet licensed for production use.
