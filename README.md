# Deception Detection via Hidden State Activation Probing in LLMs

This repository contains the code and results for a research project investigating whether Large Language Models (LLMs) develop internal representations of deception that can be detected through linear probing of hidden state activations.

## Key Findings

A linear probe trained on layer 10 hidden states of **Qwen2.5-3B-Instruct** achieves **97.5% accuracy** in distinguishing between truthful and deceptive model-generated responses across 100 diverse scenarios spanning 7 categories.

Critically, the probe identifies **white lies** (socially motivated, non-malicious falsehoods) with the same confidence as serious lies (criminal/fraudulent deception), providing strong evidence that the probe captures a **universal deception representation** rather than merely detecting malicious intent.

## Repository Structure

```
deception-probe/
├── README.md                          # This file
├── experiment_100.py                  # 100-scenario experiment script (run on Colab)
├── experiment_500.py                  # 500-scenario scaled experiment with length controls
├── results/
│   └── 100_scenario_results_report.md # Academic report of 100-scenario results
```

## Methodology

1. **Generation-based approach**: The model generates its own honest and deceptive responses to realistic scenarios (not injected text)
2. **Hidden state extraction**: Mean-pooled activations from layer 10 are extracted for each response
3. **Linear probe**: A logistic regression classifier is trained to distinguish lying vs. honest hidden states
4. **Validation**: Leave-One-Pair-Out Cross-Validation (LOPOCV), permutation tests, and TF-IDF baselines

## Results Summary (100 Scenarios)

| Metric | Value |
|--------|-------|
| Cross-Validation Accuracy | 95.0% |
| Held-Out Test Accuracy | 97.5% |
| Permutation P-value | 0.0000 |
| TF-IDF Baseline | 90.0% |
| White Lie Detection Score | 0.999 |
| Serious Lie Detection Score | 0.994 |

## Running the Experiments

### Requirements
- Google Colab with GPU (T4 or better)
- Python 3.10+

### Quick Start
1. Open Google Colab
2. Upload `experiment_100.py` or `experiment_500.py`
3. Run: `!python experiment_100.py`

The script will automatically download the model and run the full experiment pipeline.

## Categories Tested

- **Criminal**: Fraud, theft, illegal activities (20 scenarios)
- **White Lie**: Socially acceptable falsehoods (15 scenarios)
- **Workplace**: Professional misconduct (15 scenarios)
- **Relationship**: Interpersonal deception (15 scenarios)
- **Sales**: Misrepresentation of products/services (15 scenarios)
- **Academic**: Plagiarism, cheating, data fabrication (10 scenarios)
- **Health**: Medical history omission (10 scenarios)

## License

This project is for academic research purposes.
