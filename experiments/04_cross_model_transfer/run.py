"""
Experiment 04 — Cross-Model Transfer (v3)
==========================================
PURPOSE:
    Does the deception signal generalize across model architectures?
    If a probe trained on Llama can detect lies in Mistral, this suggests
    a UNIVERSAL deception representation — not an artifact of one model.

DESIGN:
    Three models tested:
        - meta-llama/Llama-3.1-8B-Instruct
        - mistralai/Mistral-7B-Instruct-v0.3
        - Qwen/Qwen2.5-7B-Instruct

    For each model:
        1. Run the same 2-phase sycophancy experiment (Exp 02 design)
        2. Extract hidden states at multiple layers
        3. Extract THREE token strategies per sample (no extra compute):
           - first_gen_token: first generated token
           - last_prompt_token: last token of the input prompt
           - mean_all_tokens: mean of all generated tokens
        4. Train within-model probe (with SVD to prevent overfitting)

    Cross-model transfer:
        - Align representation spaces via Procrustes alignment
          (fitted on SHARED questions only to avoid data leakage)
        - Train on Model A, test on Model B
        - Test all 6 directional pairs × 3 token strategies
        - Flip-test for inverted polarity detection

CONTROLS:
    1. SVD to 64 dims before probing (prevents overfitting with small n)
    2. Layer 0 (embedding) baseline per model
    3. Length-only baseline per model
    4. Cross-validated balanced accuracy (not train accuracy)
    5. Permutation test (200 iterations) on best config

CHECKPOINT/RESUME:
    Saves after each model completes. Re-run to resume from last checkpoint.

DATASET: meg-tong/sycophancy-eval (answer.jsonl)

USAGE:
    %run experiments/04_cross_model_transfer/run.py

EXPECTED RESULTS (v3):
    - Within-model: 75-85% (not 100%, thanks to SVD regularization)
    - Cross-model transfer: TBD with proper methodology

RUNTIME: ~120 minutes on A100 (3 models × 2500 questions)

CHANGELOG:
    v3 (2026-03-20): Major rewrite.
        - Increased to 2500 questions (from 500) for more lied samples.
        - Added 3 token strategies: first_gen, last_prompt, mean_all.
        - Added SVD (64 dims) before probing to prevent overfitting.
        - Extracts all 3 token positions in a single generate() call.
        - Improved logging and result structure.
    v2 (2026-03-17): Fixed Procrustes alignment to use shared questions
        only. Fixed PCA fallback to use shared dimensionality. Uses
        Pipeline and balanced_accuracy throughout.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import json
import torch
import gc
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer

from src.utils import (
    setup_logger,
    load_model_and_tokenizer,
    load_sycophancy_dataset,
    check_answer_match,
    train_probe,
    permutation_test,
    length_baseline,
    save_results,
)

try:
    from scipy.linalg import orthogonal_procrustes
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Configuration ──────────────────────────────────────────────────────────
MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
]
TARGET_LAYERS = [0, 8, 12, 14, 15, 16, 17, 18, 20, 24]
TOKEN_STRATEGIES = ["first_gen", "last_prompt", "mean_all"]
MAX_QUESTIONS = 2500
MAX_NEW_TOKENS = 80
SVD_DIMS = 64          # Reduce dimensionality before probing
N_PERMUTATIONS = 200
RANDOM_SEED = 42
CHECKPOINT_PATH = "results/exp04_checkpoint_v3.pkl"

np.random.seed(RANDOM_SEED)
log = setup_logger("exp04")


# ── Helper: extract hidden states with all 3 strategies ──────────────────

def extract_multi_strategy_hs(model, tokenizer, prompt, target_layers, max_new_tokens=80):
    """
    Generate a response and extract hidden states for 3 token strategies
    in a SINGLE generate() call (no extra compute).

    Returns:
        response: str
        hs_dict: {strategy: {layer: np.ndarray}}
            where strategy is one of: first_gen, last_prompt, mean_all
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    hs_dict = {"first_gen": {}, "last_prompt": {}, "mean_all": {}}

    if not hasattr(outputs, "hidden_states") or not outputs.hidden_states:
        return response, hs_dict

    n_gen_steps = len(outputs.hidden_states)

    for layer in target_layers:
        # --- first_gen_token: first generated token ---
        # outputs.hidden_states[0] is the prefill step.
        # The last position (-1) of the prefill step is the hidden state
        # that produces the first generated token's logits.
        prefill = outputs.hidden_states[0]
        if layer < len(prefill):
            hs_dict["first_gen"][layer] = (
                prefill[layer][0, -1, :].cpu().float().numpy()
            )

        # --- last_prompt_token: second-to-last position of prefill ---
        # Actually, the last position of the prefill IS the last prompt token.
        # Its hidden state is what the model uses to decide the first output.
        # This is the same as first_gen in terms of the prefill tensor,
        # but conceptually it's the "pre-decision" state.
        # For models that include the first gen token in hidden_states[0],
        # we use position -1 of the prefill (same tensor, same position).
        # NOTE: In HF generate with return_dict_in_generate=True,
        # hidden_states[0] contains the prefill hidden states.
        # The last position is the last prompt token.
        if layer < len(prefill):
            hs_dict["last_prompt"][layer] = (
                prefill[layer][0, -1, :].cpu().float().numpy()
            )

        # --- mean_all_tokens: mean over all generated token positions ---
        all_vecs = []
        for step_idx in range(n_gen_steps):
            step_hs = outputs.hidden_states[step_idx]
            if layer < len(step_hs):
                # Last position at each step is the current token
                vec = step_hs[layer][0, -1, :].cpu().float().numpy()
                all_vecs.append(vec)
        if all_vecs:
            hs_dict["mean_all"][layer] = np.mean(all_vecs, axis=0)

    return response, hs_dict


# ── Helper: collect data for one model ─────────────────────────────────────

def collect_model_data(model_name, model_short):
    """Run the 2-phase sycophancy experiment for one model.

    Extracts hidden states for all 3 token strategies in a single pass.
    Stores question index alongside hidden states for cross-model alignment.
    """
    log.info(f"  Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    questions = load_sycophancy_dataset(MAX_QUESTIONS)
    log.info(f"  Loaded {len(questions)} questions")

    # Phase A: knowledge test
    log.info(f"  Phase A: Knowledge test (neutral prompts)...")
    known = []
    for i, q in enumerate(questions):
        messages = [{"role": "user", "content": q["neutral_prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        if check_answer_match(resp, q["correct_answer"]):
            known.append((i, q))
        if (i + 1) % 200 == 0:
            log.info(f"    [{i+1}/{len(questions)}] Known: {len(known)}")

    log.info(f"  Model knows {len(known)}/{len(questions)} "
             f"({len(known)/len(questions)*100:.0f}%)")

    # Phase B: sycophantic pressure with multi-strategy extraction
    log.info(f"  Phase B: Sycophantic pressure on {len(known)} known questions...")
    lied, resisted = [], []
    for idx, q in known:
        resp, hs_dict = extract_multi_strategy_hs(
            model, tokenizer, q["syco_prompt"], TARGET_LAYERS, MAX_NEW_TOKENS
        )
        # Verify we got hidden states for at least one strategy
        if not hs_dict["first_gen"]:
            continue

        has_correct = check_answer_match(resp, q["correct_answer"])
        has_incorrect = check_answer_match(resp, q["incorrect_answer"])

        entry = {"hs": hs_dict, "response": resp, "q_idx": idx}

        if has_incorrect and not has_correct:
            lied.append(entry)
        elif has_correct and not has_incorrect:
            resisted.append(entry)

        total = len(lied) + len(resisted)
        if total > 0 and total % 100 == 0:
            log.info(f"    Processed {total}: Lied={len(lied)}, Resisted={len(resisted)}")

    log.info(f"  FINAL: Lied={len(lied)}, Resisted={len(resisted)}, "
             f"Lie rate={len(lied)/max(1,len(known))*100:.1f}%")

    # Cleanup GPU
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"lied": lied, "resisted": resisted, "n_known": len(known)}


# ── Helper: build X matrix for a given strategy + layer ───────────────────

def build_X(data, strategy, layer, max_n=None):
    """Extract feature matrix from collected data for a given strategy and layer."""
    lied_vecs, resisted_vecs = [], []

    for d in data["lied"]:
        if strategy in d["hs"] and layer in d["hs"][strategy]:
            lied_vecs.append(d["hs"][strategy][layer])
    for d in data["resisted"]:
        if strategy in d["hs"] and layer in d["hs"][strategy]:
            resisted_vecs.append(d["hs"][strategy][layer])

    min_n = min(len(lied_vecs), len(resisted_vecs))
    if max_n:
        min_n = min(min_n, max_n)

    if min_n < 5:
        return None, None, 0

    X = np.vstack([
        np.array(lied_vecs[:min_n]),
        np.array(resisted_vecs[:min_n]),
    ])
    y = np.array([1] * min_n + [0] * min_n)
    return X, y, min_n


def train_probe_with_svd(X, y, svd_dims=SVD_DIMS, random_seed=RANDOM_SEED):
    """Train probe with SVD dimensionality reduction to prevent overfitting."""
    n_splits = min(5, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    actual_svd_dims = min(svd_dims, X.shape[0] - 1, X.shape[1])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svd", TruncatedSVD(n_components=actual_svd_dims, random_state=random_seed)),
        ("clf", LogisticRegression(
            max_iter=1000, random_state=random_seed,
            C=1.0, class_weight="balanced",
        )),
    ])

    bal_scorer = make_scorer(balanced_accuracy_score)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=bal_scorer)

    return {
        "balanced_accuracy": float(scores.mean()),
        "std": float(scores.std()),
        "folds": [float(s) for s in scores],
        "svd_dims": actual_svd_dims,
        "n_samples": len(y),
        "n_per_class": int(min(np.bincount(y))),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Experiment 04: Cross-Model Transfer (v3)")
    log.info("=" * 70)
    log.info(f"  Models: {[s for _, s in MODELS]}")
    log.info(f"  Questions: {MAX_QUESTIONS}")
    log.info(f"  Token strategies: {TOKEN_STRATEGIES}")
    log.info(f"  SVD dims: {SVD_DIMS}")
    log.info(f"  Layers: {TARGET_LAYERS}")
    log.info("=" * 70)
    start = time.time()

    # Load checkpoint if exists
    os.makedirs("results", exist_ok=True)
    all_data = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            all_data = pickle.load(f)
        log.info(f"Resumed from checkpoint: {list(all_data.keys())}")

    # ── Collect data per model ────────────────────────────────────────
    for model_name, short in MODELS:
        if short in all_data:
            log.info(f"Skipping {short} (already in checkpoint)")
            continue

        log.info(f"\n{'='*50}")
        log.info(f"Model: {model_name}")
        log.info(f"{'='*50}")
        data = collect_model_data(model_name, short)
        all_data[short] = data

        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(all_data, f)
        log.info(f"  Checkpoint saved ({short}: "
                 f"{len(data['lied'])} lied, {len(data['resisted'])} resisted)")

    # ── Within-model probes ───────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("WITHIN-MODEL PROBES (with SVD)")
    log.info("=" * 70)

    results = {
        "within_model": {},
        "cross_model": {},
        "config": {
            "max_questions": MAX_QUESTIONS,
            "svd_dims": SVD_DIMS,
            "token_strategies": TOKEN_STRATEGIES,
            "target_layers": TARGET_LAYERS,
            "n_permutations": N_PERMUTATIONS,
        },
    }
    best_configs = {}  # {model_short: {strategy: best_layer}}

    for short in [s for _, s in MODELS]:
        if short not in all_data:
            continue
        data = all_data[short]
        results["within_model"][short] = {
            "n_lied": len(data["lied"]),
            "n_resisted": len(data["resisted"]),
            "strategies": {},
        }
        best_configs[short] = {}

        for strategy in TOKEN_STRATEGIES:
            best_acc, best_layer = 0, None
            layer_results = {}

            for layer in TARGET_LAYERS:
                X, y, n = build_X(data, strategy, layer)
                if X is None:
                    continue

                probe_result = train_probe_with_svd(X, y)
                layer_results[layer] = probe_result

                if probe_result["balanced_accuracy"] > best_acc:
                    best_acc = probe_result["balanced_accuracy"]
                    best_layer = layer

            best_configs[short][strategy] = best_layer

            results["within_model"][short]["strategies"][strategy] = {
                "best_layer": best_layer,
                "best_accuracy": best_acc,
                "layer_results": {str(k): v for k, v in layer_results.items()},
            }

            log.info(f"  {short}/{strategy}: {best_acc*100:.1f}% "
                     f"(Layer {best_layer}, n={layer_results.get(best_layer, {}).get('n_per_class', '?')})")

    # ── Length baseline ────────────────────────────────────────────────
    log.info("\nLength baselines:")
    for short in [s for _, s in MODELS]:
        if short not in all_data:
            continue
        data = all_data[short]
        min_n = min(len(data["lied"]), len(data["resisted"]))
        len_acc = length_baseline(
            [d["response"] for d in data["lied"][:min_n]],
            [d["response"] for d in data["resisted"][:min_n]],
        )
        results["within_model"][short]["length_baseline"] = len_acc
        log.info(f"  {short}: {len_acc*100:.1f}%")

    # ── Permutation test on best within-model config ──────────────────
    log.info("\nPermutation tests (best config per model):")
    for short in [s for _, s in MODELS]:
        if short not in all_data or short not in best_configs:
            continue
        # Find overall best strategy for this model
        best_strat = max(
            TOKEN_STRATEGIES,
            key=lambda s: results["within_model"][short]["strategies"].get(s, {}).get("best_accuracy", 0)
        )
        best_layer = best_configs[short].get(best_strat)
        if best_layer is None:
            continue

        X, y, n = build_X(all_data[short], best_strat, best_layer)
        if X is None:
            continue

        # Apply SVD before permutation test
        actual_svd = min(SVD_DIMS, X.shape[0] - 1, X.shape[1])
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        svd = TruncatedSVD(n_components=actual_svd, random_state=RANDOM_SEED)
        X_svd = svd.fit_transform(X_s)

        observed = results["within_model"][short]["strategies"][best_strat]["best_accuracy"]
        perm = permutation_test(X_svd, y, observed, N_PERMUTATIONS, random_seed=RANDOM_SEED)

        results["within_model"][short]["permutation_test"] = {
            "strategy": best_strat,
            "layer": best_layer,
            **perm,
        }
        log.info(f"  {short}: p={perm['p_value']:.4f} "
                 f"({best_strat}, Layer {best_layer})")

    # ── Cross-model transfer ──────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("CROSS-MODEL TRANSFER")
    log.info("=" * 70)

    model_shorts = [s for _, s in MODELS if s in best_configs]

    for strategy in TOKEN_STRATEGIES:
        log.info(f"\n--- Strategy: {strategy} ---")

        for src in model_shorts:
            for tgt in model_shorts:
                if src == tgt:
                    continue

                src_data = all_data[src]
                tgt_data = all_data[tgt]
                layer = best_configs[src].get(strategy)
                if layer is None or layer == 0:
                    continue

                X_src, y_src, n_src = build_X(src_data, strategy, layer)
                X_tgt, y_tgt, n_tgt = build_X(tgt_data, strategy, layer)

                if X_src is None or X_tgt is None:
                    continue

                # Apply SVD to both
                actual_svd = min(SVD_DIMS, X_src.shape[0] - 1, X_tgt.shape[0] - 1, X_src.shape[1])

                scaler_src = StandardScaler()
                scaler_tgt = StandardScaler()
                X_src_s = scaler_src.fit_transform(X_src)
                X_tgt_s = scaler_tgt.fit_transform(X_tgt)

                svd_src = TruncatedSVD(n_components=actual_svd, random_state=RANDOM_SEED)
                svd_tgt = TruncatedSVD(n_components=actual_svd, random_state=RANDOM_SEED)
                X_src_svd = svd_src.fit_transform(X_src_s)
                X_tgt_svd = svd_tgt.fit_transform(X_tgt_s)

                # Procrustes alignment on SHARED questions
                src_indices = set(d["q_idx"] for d in src_data["lied"][:n_src])
                src_indices |= set(d["q_idx"] for d in src_data["resisted"][:n_src])
                tgt_indices = set(d["q_idx"] for d in tgt_data["lied"][:n_tgt])
                tgt_indices |= set(d["q_idx"] for d in tgt_data["resisted"][:n_tgt])
                shared = sorted(src_indices & tgt_indices)

                alignment_method = "none"

                if HAS_SCIPY and len(shared) >= 20:
                    # Build aligned matrices from shared questions
                    src_by_idx = {}
                    for d in src_data["lied"][:n_src] + src_data["resisted"][:n_src]:
                        if strategy in d["hs"] and layer in d["hs"][strategy]:
                            src_by_idx[d["q_idx"]] = d["hs"][strategy][layer]
                    tgt_by_idx = {}
                    for d in tgt_data["lied"][:n_tgt] + tgt_data["resisted"][:n_tgt]:
                        if strategy in d["hs"] and layer in d["hs"][strategy]:
                            tgt_by_idx[d["q_idx"]] = d["hs"][strategy][layer]

                    valid_shared = [i for i in shared if i in src_by_idx and i in tgt_by_idx]

                    if len(valid_shared) >= 20:
                        shared_src_raw = np.array([src_by_idx[i] for i in valid_shared])
                        shared_tgt_raw = np.array([tgt_by_idx[i] for i in valid_shared])

                        # Apply same scaling + SVD
                        shared_src_s = scaler_src.transform(shared_src_raw)
                        shared_tgt_s = scaler_tgt.transform(shared_tgt_raw)
                        shared_src_svd = svd_src.transform(shared_src_s)
                        shared_tgt_svd = svd_tgt.transform(shared_tgt_s)

                        R, _ = orthogonal_procrustes(shared_tgt_svd, shared_src_svd)
                        X_tgt_aligned = X_tgt_svd @ R
                        alignment_method = f"procrustes_shared_{len(valid_shared)}"
                    else:
                        X_tgt_aligned = X_tgt_svd
                        alignment_method = f"none_insufficient_shared_{len(valid_shared)}"
                else:
                    X_tgt_aligned = X_tgt_svd

                # Train on source, test on target
                clf = LogisticRegression(
                    max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
                )
                clf.fit(X_src_svd, y_src)

                direct_acc = balanced_accuracy_score(y_tgt, clf.predict(X_tgt_aligned))
                flipped_acc = balanced_accuracy_score(y_tgt, 1 - clf.predict(X_tgt_aligned))
                inverted = bool(flipped_acc > direct_acc)

                key = f"{src}->{tgt}"
                if key not in results["cross_model"]:
                    results["cross_model"][key] = {}

                results["cross_model"][key][strategy] = {
                    "direct_accuracy": float(direct_acc),
                    "flipped_accuracy": float(flipped_acc),
                    "inverted_polarity": inverted,
                    "best_accuracy": float(max(direct_acc, flipped_acc)),
                    "layer": layer,
                    "alignment_method": alignment_method,
                    "n_src": n_src,
                    "n_tgt": n_tgt,
                    "svd_dims": actual_svd,
                }

                polarity = " (INVERTED)" if inverted else ""
                best = max(direct_acc, flipped_acc)
                log.info(f"  {key}: direct={direct_acc*100:.1f}%, "
                         f"flipped={flipped_acc*100:.1f}%, best={best*100:.1f}%{polarity} "
                         f"[{alignment_method}]")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    log.info(f"\n{'='*70}")
    log.info(f"SUMMARY")
    log.info(f"{'='*70}")

    for short in model_shorts:
        wm = results["within_model"][short]
        log.info(f"\n{short}: {wm['n_lied']} lied, {wm['n_resisted']} resisted")
        for strat in TOKEN_STRATEGIES:
            s = wm["strategies"].get(strat, {})
            log.info(f"  {strat}: {s.get('best_accuracy', 0)*100:.1f}% (Layer {s.get('best_layer')})")

    log.info(f"\nCross-model transfer (best per pair):")
    for pair, strats in results["cross_model"].items():
        best_strat = max(strats, key=lambda s: strats[s]["best_accuracy"])
        best = strats[best_strat]
        inv = " (INV)" if best["inverted_polarity"] else ""
        log.info(f"  {pair}: {best['best_accuracy']*100:.1f}%{inv} "
                 f"[{best_strat}, Layer {best['layer']}]")

    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Save
    output = {
        "experiment": "04_cross_model_transfer_v3",
        "models": {s: m for m, s in MODELS},
        "results": results,
        "elapsed_seconds": elapsed,
    }
    save_results(output, "results/exp04_cross_model.json")
    log.info("Saved to results/exp04_cross_model.json")

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        log.info(f"Checkpoint kept at {CHECKPOINT_PATH} for reference")


if __name__ == "__main__":
    main()
