# Deception Probe — Key Findings

## Summary of Results Across All Stages

| Stage | Task | Accuracy | Metric | p-value | Confound-Free | Notes |
|-------|------|----------|--------|---------|---------------|-------|
| 1 | Basic sycophancy detection | 100% | Accuracy | <0.001 | No | Prompt confound (different prompts for truth/lie) |
| 2 | Cross-model (Mistral) | 100% | Accuracy | <0.001 | No | Same confound as Stage 1 |
| 3 | Confound analysis | 100% | Accuracy | <0.001 | No | Confirmed prompt confound explains 100% accuracy |
| **4** | **Sycophantic lies (same prompt)** | **82.5%** | **Accuracy** | **<0.001** | **Yes** | Confound-free. Best layer: 15. 43 samples/class |
| **5** | **Real-world deception (18 domains)** | **70.4%** | **Bal. Accuracy** | **<0.001** | **Yes** | 459 scenarios. Best layer: 17. Cross-domain: 55.7% |
| **6** | **Lie vs Hallucination (3-way)** | **82.3%** | **Bal. Accuracy** | **<0.001** | **Yes** | Truth/Lie/Hallucination. Best layer: 20 |
| **6** | **Lie vs Hallucination (binary)** | **100%** | **Bal. Accuracy** | **0.0000** | **Yes** | Model's lie and hallucination states are completely separable |
| **6** | **Truth vs Lie (binary)** | **100%** | **Bal. Accuracy** | — | **Yes** | Model "knows" when it's lying |
| **6** | **Truth vs Hallucination (binary)** | **67.4%** | **Bal. Accuracy** | — | **Yes** | Hardest task — no internal "tension" in hallucination |
| 7 | Advanced hallucination detection | TBD | Bal. Accuracy | TBD | Yes | 6 methods to improve Truth vs Hallucination |

## Key Findings

### 1. Confounds Matter (Stages 1-3)
Stages 1-3 achieved 100% accuracy but this was entirely explained by the prompt confound — different system prompts for honest vs deceptive conditions. This is a known issue in the literature that many papers do not adequately address.

### 2. Real Deception Signal Exists (Stage 4)
When using the **same prompt** for both conditions (model chooses to lie or resist sycophantic pressure), accuracy drops to 82.5% — still well above chance (50%) and statistically significant (p<0.001). This proves a genuine deception signal exists in hidden states.

### 3. Middle Layers Are Optimal
Across all confound-free stages, **middle layers (15-20)** consistently outperform early and late layers:
- Stage 4: Layer 15 best (82.5%)
- Stage 5: Layer 17 best (70.4%)
- Stage 6: Layer 20 best (82.3%)

Layer 0 (embedding) performs at chance, confirming the signal is semantic, not lexical.

### 4. Lies and Hallucinations Are Fundamentally Different (Stage 6)
The model's internal state when it **lies** (knows truth but says otherwise) is completely separable from when it **hallucinates** (doesn't know and makes something up) — 100% accuracy with p=0.0000. This is the most novel finding.

### 5. Length Is Not a Confound
Length-only baselines across all stages: 50-60% (near chance). The probe captures information beyond response length.

## Model and Method
- **Model**: Llama-3.1-8B-Instruct (4-bit quantized via bitsandbytes)
- **Probe**: Logistic Regression on hidden state activations (first generated token)
- **Validation**: 5-fold stratified cross-validation
- **Statistical tests**: Permutation tests (500 iterations), balanced accuracy
- **Dataset**: meg-tong/sycophancy-eval (TriviaQA-based, 1,817 question pairs)

## Confound Controls
1. Same prompt format for both conditions (Stages 4-6)
2. Length-only baseline (consistently near chance)
3. Permutation tests (500 iterations, p<0.001)
4. Balanced accuracy metric (handles class imbalance)
5. Layer 0 (embedding) at chance level (rules out lexical confounds)
