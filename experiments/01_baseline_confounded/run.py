"""
Experiment 01 — Baseline Detection (Confounded)
================================================
PURPOSE:
    Demonstrate that a linear probe can detect sycophantic lies at ~100%
    accuracy — but this result is MISLEADING because the prompts differ
    between truth and lie conditions (the "prompt confound").

    This experiment exists to motivate the confound-free design in Exp 02.

DESIGN:
    - Truth condition:  neutral prompt → model answers correctly
    - Lie condition:    sycophantic prompt → model agrees with wrong answer
    - Probe: Logistic Regression on hidden states (first generated token)
    - Layers: 0 (embedding), 8, 12, 16, 24

WHY 100% IS NOT IMPRESSIVE HERE:
    The two conditions use DIFFERENT prompts, so the probe may simply
    learn to distinguish the prompt format rather than the lie itself.
    This is confirmed by the Layer 0 (embedding) baseline also reaching
    ~100%, meaning the signal is lexical, not semantic.

DATASET: meg-tong/sycophancy-eval (answer.jsonl) — TriviaQA-based
MODEL:   meta-llama/Llama-3.1-8B-Instruct (4-bit quantized)

USAGE (Google Colab with GPU):
    !pip install -q transformers accelerate bitsandbytes datasets scikit-learn
    %cd /content/deception-probe
    %run experiments/01_baseline_confounded/run.py

EXPECTED RESULTS:
    ~100% accuracy across all layers (including Layer 0)
    This confirms the prompt confound — the probe detects the prompt, not the lie.

RUNTIME: ~15-25 minutes on A100
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import time

from src.utils import (
    setup_logger,
    load_model_and_tokenizer,
    load_sycophancy_dataset,
    extract_hidden_states,
    check_answer_match,
    train_probe,
    permutation_test,
    length_baseline,
    save_results,
)

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TARGET_LAYERS = [0, 8, 12, 16, 24]
MAX_QUESTIONS = 250
MAX_NEW_TOKENS = 80
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
log = setup_logger("exp01")

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 01: Baseline Detection (Confounded)")
    log.info("=" * 60)
    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Layers: {TARGET_LAYERS}")
    log.info(f"Max questions: {MAX_QUESTIONS}")
    start = time.time()

    # 1. Load dataset
    log.info("Loading dataset...")
    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"  {len(questions)} question pairs loaded")

    # 2. Load model
    log.info(f"Loading {MODEL_NAME}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # 3. Generate responses and extract hidden states
    log.info("Generating responses (neutral + sycophantic)...")
    truth_data, lie_data = [], []

    for i, q in enumerate(questions):
        # Truth: neutral prompt
        resp_truth, hs_truth = extract_hidden_states(
            model, tokenizer, q["neutral_prompt"], TARGET_LAYERS, MAX_NEW_TOKENS
        )
        if hs_truth and len(hs_truth) == len(TARGET_LAYERS):
            if check_answer_match(resp_truth, q["correct_answer"]):
                truth_data.append({"hs": hs_truth, "response": resp_truth})

        # Lie: sycophantic prompt
        resp_syco, hs_syco = extract_hidden_states(
            model, tokenizer, q["syco_prompt"], TARGET_LAYERS, MAX_NEW_TOKENS
        )
        if hs_syco and len(hs_syco) == len(TARGET_LAYERS):
            has_incorrect = check_answer_match(resp_syco, q["incorrect_answer"])
            has_correct = check_answer_match(resp_syco, q["correct_answer"])
            if has_incorrect and not has_correct:
                lie_data.append({"hs": hs_syco, "response": resp_syco})

        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(questions)}] Truth: {len(truth_data)}, Lies: {len(lie_data)}")

    log.info(f"  Final: {len(truth_data)} truth, {len(lie_data)} lies")

    # 4. Train probes
    min_n = min(len(truth_data), len(lie_data))
    if min_n < 5:
        log.error(f"Not enough samples ({min_n}). Aborting.")
        return

    log.info(f"Training probes ({min_n} per class)...")
    results = {}

    for layer in TARGET_LAYERS:
        X_truth = np.array([d["hs"][layer] for d in truth_data[:min_n]])
        X_lie = np.array([d["hs"][layer] for d in lie_data[:min_n]])
        X = np.vstack([X_truth, X_lie])
        y = np.array([0] * min_n + [1] * min_n)

        probe_result = train_probe(X, y, random_seed=RANDOM_SEED)
        perm_result = permutation_test(X, y, probe_result["balanced_accuracy"],
                                        n_permutations=200, random_seed=RANDOM_SEED)

        results[layer] = {**probe_result, **perm_result}
        tag = " (EMBEDDING)" if layer == 0 else ""
        sig = "***" if perm_result["p_value"] < 0.001 else ""
        log.info(f"  Layer {layer:2d}: {probe_result['balanced_accuracy']*100:.1f}% "
                 f"(p={perm_result['p_value']:.4f}) {sig}{tag}")

    # Length baseline
    len_acc = length_baseline(
        [d["response"] for d in truth_data[:min_n]],
        [d["response"] for d in lie_data[:min_n]],
    )
    log.info(f"  Length-only baseline: {len_acc*100:.1f}%")

    # 5. Save
    output = {
        "experiment": "01_baseline_confounded",
        "description": "Baseline sycophancy detection with prompt confound (different prompts)",
        "model": MODEL_NAME,
        "layers": TARGET_LAYERS,
        "n_truth": len(truth_data),
        "n_lies": len(lie_data),
        "n_balanced": min_n,
        "length_baseline": len_acc,
        "results_per_layer": {str(k): v for k, v in results.items()},
        "conclusion": "~100% accuracy INCLUDING Layer 0 confirms prompt confound. "
                       "The probe detects the prompt format, not the lie itself.",
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp01_baseline.json")
    log.info("Results saved to results/exp01_baseline.json")
    log.info("=" * 60)
    log.info("CONCLUSION: High accuracy here is MISLEADING — see Experiment 02")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
