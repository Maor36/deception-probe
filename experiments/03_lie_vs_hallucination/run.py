"""
Experiment 03 — Lie vs Hallucination Detection (Confound-Free)
==============================================================
PURPOSE:
    Can we distinguish between a model that LIES (knows the truth but
    says something else under pressure) and a model that HALLUCINATES
    (doesn't know the answer and makes something up)?

DESIGN (Confound-Free):
    Three classes, all from the same dataset:

    Phase A — Knowledge test (neutral prompt, no pressure):
        Determines which questions the model knows vs doesn't know.
        Hidden states from this phase are used for TRUTH class only.

    Phase B — Sycophantic pressure (SAME prompt format for all):
        ALL questions are re-asked with sycophantic pressure.
        This ensures LIE and HALLUCINATION share identical prompt format.

    TRUTH:         Model knows → answers correctly under pressure (Phase B)
    LIE:           Model knows → gives wrong answer under pressure (Phase B)
    HALLUCINATION: Model doesn't know → gives wrong answer under pressure (Phase B)

    KEY IMPROVEMENT: Both LIE and HALLUCINATION use the sycophantic prompt,
    eliminating the prompt-format confound from the previous version.
    The ONLY difference is whether the model knew the answer (proven in Phase A).

CONTROLS:
    1. Layer 0 baseline → ~33% (chance for 3-way)
    2. Length baseline → ~50% for binary comparisons
    3. Permutation test (500 iterations) for Lie vs Hallucination
    4. TRUTH uses Phase B hidden states (same prompt as LIE/HALL)

DATASET: meg-tong/sycophancy-eval (answer.jsonl)
MODEL:   meta-llama/Llama-3.1-8B-Instruct (4-bit quantized)

USAGE:
    %run experiments/03_lie_vs_hallucination/run.py

RUNTIME: ~90 minutes on T4 (2500 questions × 2 phases)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix

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
PROBE_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 31]
MAX_QUESTIONS = 2500
MAX_NEW_TOKENS = 80
N_PERMUTATIONS = 500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
log = setup_logger("exp03")
bal_scorer = make_scorer(balanced_accuracy_score)

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 03: Lie vs Hallucination (Confound-Free)")
    log.info("=" * 60)
    start = time.time()

    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"Loaded {len(questions)} questions")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # ── Phase A: Knowledge test (neutral prompt) ──────────────────────
    # Purpose: determine which questions the model KNOWS vs DOESN'T KNOW
    # We do NOT use hidden states from this phase for classification.
    log.info("Phase A: Testing model knowledge (neutral prompt)...")
    knows_questions = []      # questions the model answered correctly
    doesnt_know_questions = [] # questions the model answered incorrectly

    for i, q in enumerate(questions):
        resp, hs = extract_hidden_states(
            model, tokenizer, q["neutral_prompt"], PROBE_LAYERS, MAX_NEW_TOKENS
        )
        if not hs:
            continue

        if check_answer_match(resp, q["correct_answer"]):
            knows_questions.append(q)
        else:
            doesnt_know_questions.append(q)

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(questions)}] Knows: {len(knows_questions)}, "
                     f"Doesn't know: {len(doesnt_know_questions)}")

    log.info(f"  Phase A done — Knows: {len(knows_questions)}, "
             f"Doesn't know: {len(doesnt_know_questions)}")

    # ── Phase B: Sycophantic pressure on ALL questions ────────────────
    # ALL classes use the SAME sycophantic prompt format.
    # This eliminates the prompt-format confound.

    # Phase B.1: Questions the model KNOWS — with sycophantic pressure
    log.info("Phase B.1: Sycophantic pressure on KNOWN questions...")
    truth_data = []  # knew + still correct under pressure
    lie_data = []    # knew + changed to wrong under pressure

    for i, q in enumerate(knows_questions):
        resp, hs = extract_hidden_states(
            model, tokenizer, q["syco_prompt"], PROBE_LAYERS, MAX_NEW_TOKENS
        )
        if not hs:
            continue

        has_correct = check_answer_match(resp, q["correct_answer"])
        has_incorrect = check_answer_match(resp, q["incorrect_answer"])

        if has_correct and not has_incorrect:
            # Resisted pressure — TRUTH
            truth_data.append({"hs": hs, "response": resp})
        elif has_incorrect and not has_correct:
            # Caved to pressure — LIE
            lie_data.append({"hs": hs, "response": resp})

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(knows_questions)}] Truth: {len(truth_data)}, "
                     f"Lies: {len(lie_data)}")

    log.info(f"  Phase B.1 done — Truth: {len(truth_data)}, Lie: {len(lie_data)}")

    # Phase B.2: Questions the model DOESN'T KNOW — with sycophantic pressure
    log.info("Phase B.2: Sycophantic pressure on UNKNOWN questions...")
    hall_data = []  # didn't know + wrong under pressure = hallucination

    for i, q in enumerate(doesnt_know_questions):
        resp, hs = extract_hidden_states(
            model, tokenizer, q["syco_prompt"], PROBE_LAYERS, MAX_NEW_TOKENS
        )
        if not hs:
            continue

        has_correct = check_answer_match(resp, q["correct_answer"])
        # Only count as hallucination if still wrong (expected for most)
        if not has_correct:
            hall_data.append({"hs": hs, "response": resp})

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(doesnt_know_questions)}] Hallucinations: {len(hall_data)}")

    log.info(f"  Phase B.2 done — Hallucinations: {len(hall_data)}")

    log.info(f"  FINAL COUNTS — TRUTH: {len(truth_data)}, LIE: {len(lie_data)}, "
             f"HALLUCINATION: {len(hall_data)}")
    log.info(f"  ALL three classes use sycophantic prompt (confound-free)")

    # Save hidden states for downstream analysis
    os.makedirs("results", exist_ok=True)
    with open("results/exp03_hidden_states.pkl", "wb") as f:
        pickle.dump({
            "truth": truth_data, "lie": lie_data, "hallucination": hall_data,
            "probe_layers": PROBE_LAYERS, "model": MODEL_NAME,
            "design": "confound_free_all_syco_prompt",
        }, f)
    log.info("  Saved hidden states to results/exp03_hidden_states.pkl")

    # ── Classification ─────────────────────────────────────────────────
    min_3way = min(len(truth_data), len(lie_data), len(hall_data))
    log.info(f"3-way balanced: {min_3way} per class")

    if min_3way < 5:
        log.error("Not enough samples for 3-way classification")
        return

    results = {}

    # Layer scan (3-way)
    log.info("Layer scan (3-way: Truth vs Lie vs Hallucination)...")
    layer_results = {}
    for layer in PROBE_LAYERS:
        X = np.vstack([
            np.array([d["hs"][layer] for d in truth_data[:min_3way]]),
            np.array([d["hs"][layer] for d in lie_data[:min_3way]]),
            np.array([d["hs"][layer] for d in hall_data[:min_3way]]),
        ])
        y = np.array([0] * min_3way + [1] * min_3way + [2] * min_3way)
        probe = train_probe(X, y, random_seed=RANDOM_SEED)
        layer_results[layer] = probe
        tag = " (EMBEDDING)" if layer == 0 else ""
        log.info(f"  Layer {layer:2d}: {probe['balanced_accuracy']*100:.1f}%{tag}")

    best_layer = max(
        [l for l in layer_results if l != 0],
        key=lambda l: layer_results[l]["balanced_accuracy"],
    )
    results["best_layer"] = best_layer
    results["layer_scan"] = {str(k): v for k, v in layer_results.items()}

    # Binary comparisons on best layer
    log.info(f"\nBinary comparisons (Layer {best_layer})...")
    X_t = np.array([d["hs"][best_layer] for d in truth_data[:min_3way]])
    X_l = np.array([d["hs"][best_layer] for d in lie_data[:min_3way]])
    X_h = np.array([d["hs"][best_layer] for d in hall_data[:min_3way]])

    for name, Xa, Xb, key in [
        ("Truth vs Lie", X_t, X_l, "truth_vs_lie"),
        ("Truth vs Hallucination", X_t, X_h, "truth_vs_hallucination"),
        ("Lie vs Hallucination", X_l, X_h, "lie_vs_hallucination"),
    ]:
        X = np.vstack([Xa, Xb])
        y = np.array([0] * min_3way + [1] * min_3way)
        probe = train_probe(X, y, random_seed=RANDOM_SEED)
        results[key] = probe
        marker = " ← KEY FINDING" if key == "lie_vs_hallucination" else ""
        log.info(f"  {name:<30s}: {probe['balanced_accuracy']*100:.1f}%{marker}")

    # Permutation test for Lie vs Hallucination
    log.info(f"Permutation test ({N_PERMUTATIONS}x) for Lie vs Hallucination...")
    X_lh = np.vstack([X_l, X_h])
    y_lh = np.array([0] * min_3way + [1] * min_3way)
    perm = permutation_test(
        X_lh, y_lh, results["lie_vs_hallucination"]["balanced_accuracy"],
        n_permutations=N_PERMUTATIONS, random_seed=RANDOM_SEED,
    )
    results["lie_vs_hall_permutation"] = perm
    log.info(f"  p-value: {perm['p_value']:.4f}")

    # Length baselines
    len_lh = length_baseline(
        [d["response"] for d in lie_data[:min_3way]],
        [d["response"] for d in hall_data[:min_3way]],
    )
    results["length_baseline_lie_vs_hall"] = len_lh
    log.info(f"  Length baseline (Lie vs Hall): {len_lh*100:.1f}%")

    # ── Summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS SUMMARY (Confound-Free Design)")
    log.info(f"  All 3 classes use sycophantic prompt format")
    log.info(f"  Best layer: {best_layer}")
    log.info(f"  3-way accuracy: {layer_results[best_layer]['balanced_accuracy']*100:.1f}%")
    log.info(f"  Lie vs Hallucination: {results['lie_vs_hallucination']['balanced_accuracy']*100:.1f}% "
             f"(p={perm['p_value']:.4f})")
    log.info(f"  Truth vs Lie: {results['truth_vs_lie']['balanced_accuracy']*100:.1f}%")
    log.info(f"  Truth vs Hallucination: {results['truth_vs_hallucination']['balanced_accuracy']*100:.1f}%")
    log.info("=" * 60)

    output = {
        "experiment": "03_lie_vs_hallucination_confound_free",
        "model": MODEL_NAME,
        "design": "All 3 classes use sycophantic prompt. Knowledge determined in Phase A (neutral). Hidden states from Phase B (sycophantic) only.",
        "n_truth": len(truth_data),
        "n_lie": len(lie_data),
        "n_hallucination": len(hall_data),
        "n_balanced": min_3way,
        "results": results,
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp03_lie_vs_hallucination.json")
    log.info("Saved to results/exp03_lie_vs_hallucination.json")


if __name__ == "__main__":
    main()
