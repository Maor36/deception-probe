"""
DECEPTION PROBE — Experiment 08, Step 3: Probe Training & Analysis
====================================================================

PURPOSE:
    Train a linear probe on the hidden states collected in Step 1,
    using the labels from Step 2, to detect when the model conceals
    information (lies) vs. discloses it (truth).

    This is the SAME probing methodology as Experiment 02, but applied
    to real-world professional scenarios instead of trivia questions.

ANALYSES:
    1. Per-layer probe accuracy (which layer best separates truth/lie?)
    2. Permutation test (p-value for statistical significance)
    3. Length baseline (is it just response length?)
    4. Per-domain accuracy (is medical deception easier to detect than sales?)
    5. Cross-domain transfer (train on sales, test on medical — does it generalize?)

USAGE:
    python experiments/08_realworld_deception/step3_probe.py \
        --labels results/exp08_labeled.json \
        --hidden results/exp08_hidden_states.npz
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import (
    setup_logger,
    train_probe,
    permutation_test,
    save_results,
)

logger = setup_logger("exp08_probe")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(labels_path: str, hidden_path: str):
    """Load labeled responses and corresponding hidden states."""
    with open(labels_path, "r") as f:
        data = json.load(f)

    responses = data["responses"]
    hidden = np.load(hidden_path)

    # Filter to only labeled entries (label = 0 or 1, not -1)
    valid_indices = []
    labels = []
    domains = []
    lengths = []

    for i, entry in enumerate(responses):
        if entry.get("label") in [0, 1]:
            valid_indices.append(i)
            labels.append(entry["label"])
            domains.append(entry["domain"])
            lengths.append(entry.get("response_length", len(entry.get("response", ""))))

    labels = np.array(labels)
    domains = np.array(domains)
    lengths = np.array(lengths)

    logger.info(f"Valid labeled entries: {len(valid_indices)}")
    logger.info(f"  Disclosed (1): {np.sum(labels == 1)}")
    logger.info(f"  Concealed (0): {np.sum(labels == 0)}")

    # Extract hidden states for valid indices only
    layer_keys = sorted(
        [k for k in hidden.files if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1])
    )

    hidden_states = {}
    for key in layer_keys:
        layer_idx = int(key.split("_")[1])
        all_vectors = hidden[key]
        hidden_states[layer_idx] = all_vectors[valid_indices]

    return hidden_states, labels, domains, lengths


# ---------------------------------------------------------------------------
# Per-layer probe
# ---------------------------------------------------------------------------
def run_per_layer_probe(hidden_states: dict, labels: np.ndarray):
    """Train probe at each layer and find the best one."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    results = {}
    best_layer = -1
    best_acc = 0.0

    for layer_idx in sorted(hidden_states.keys()):
        X = hidden_states[layer_idx]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        scores = cross_val_score(
            pipe, X, labels,
            cv=min(5, np.min(np.bincount(labels))),
            scoring="balanced_accuracy",
        )
        mean_acc = scores.mean()
        std_acc = scores.std()

        results[layer_idx] = {"mean": mean_acc, "std": std_acc}

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_layer = layer_idx

        if layer_idx % 4 == 0 or layer_idx == max(hidden_states.keys()):
            logger.info(f"  Layer {layer_idx:2d}: {mean_acc:.3f} ± {std_acc:.3f}")

    logger.info(f"\n  BEST LAYER: {best_layer} ({best_acc:.3f})")
    return results, best_layer


# ---------------------------------------------------------------------------
# Length baseline
# ---------------------------------------------------------------------------
def run_length_baseline(lengths: np.ndarray, labels: np.ndarray):
    """Check if response length alone predicts disclosure."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = lengths.reshape(-1, 1)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    scores = cross_val_score(
        pipe, X, labels,
        cv=min(5, np.min(np.bincount(labels))),
        scoring="balanced_accuracy",
    )
    mean_acc = scores.mean()
    logger.info(f"  Length baseline: {mean_acc:.3f} (should be ~0.50)")
    return mean_acc


# ---------------------------------------------------------------------------
# Per-domain analysis
# ---------------------------------------------------------------------------
def run_per_domain_analysis(hidden_states: dict, labels: np.ndarray,
                            domains: np.ndarray, best_layer: int):
    """Check probe accuracy per domain."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = hidden_states[best_layer]
    unique_domains = np.unique(domains)
    domain_results = {}

    for domain in unique_domains:
        mask = domains == domain
        n_samples = mask.sum()
        n_pos = labels[mask].sum()
        n_neg = n_samples - n_pos

        # Need at least 2 of each class for CV
        if n_pos < 2 or n_neg < 2:
            domain_results[domain] = {
                "n": int(n_samples),
                "disclosed": int(n_pos),
                "concealed": int(n_neg),
                "accuracy": None,
                "note": "too few samples for CV",
            }
            continue

        X_domain = X[mask]
        y_domain = labels[mask]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        n_folds = min(5, min(n_pos, n_neg))
        scores = cross_val_score(
            pipe, X_domain, y_domain,
            cv=n_folds,
            scoring="balanced_accuracy",
        )

        domain_results[domain] = {
            "n": int(n_samples),
            "disclosed": int(n_pos),
            "concealed": int(n_neg),
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
        }

        logger.info(
            f"  {domain:35s} | n={n_samples:3d} | "
            f"acc={scores.mean():.3f} | "
            f"disclosed={n_pos}, concealed={n_neg}"
        )

    return domain_results


# ---------------------------------------------------------------------------
# Cross-domain transfer
# ---------------------------------------------------------------------------
def run_cross_domain_transfer(hidden_states: dict, labels: np.ndarray,
                               domains: np.ndarray, best_layer: int):
    """Train on some domains, test on others."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    X = hidden_states[best_layer]
    unique_domains = np.unique(domains)

    # Only use domains with enough samples
    valid_domains = []
    for d in unique_domains:
        mask = domains == d
        n_pos = labels[mask].sum()
        n_neg = mask.sum() - n_pos
        if n_pos >= 3 and n_neg >= 3:
            valid_domains.append(d)

    if len(valid_domains) < 2:
        logger.info("  Not enough valid domains for cross-domain transfer")
        return {}

    # Leave-one-domain-out
    transfer_results = {}
    for test_domain in valid_domains:
        train_mask = np.array([d != test_domain for d in domains])
        test_mask = domains == test_domain

        # Check train set has both classes
        if len(np.unique(labels[train_mask])) < 2:
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        pipe.fit(X[train_mask], labels[train_mask])
        preds = pipe.predict(X[test_mask])
        acc = balanced_accuracy_score(labels[test_mask], preds)

        transfer_results[test_domain] = {
            "accuracy": float(acc),
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
        }

        logger.info(
            f"  Train: all except {test_domain:30s} → "
            f"Test: {test_domain} | acc={acc:.3f}"
        )

    if transfer_results:
        mean_transfer = np.mean([r["accuracy"] for r in transfer_results.values()])
        logger.info(f"\n  Mean cross-domain transfer: {mean_transfer:.3f}")

    return transfer_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train deception probe on real-world scenarios")
    parser.add_argument("--labels", type=str, default="results/exp08_labeled.json")
    parser.add_argument("--hidden", type=str, default="results/exp08_hidden_states.npz")
    args = parser.parse_args()

    labels_path = os.path.join(REPO_ROOT, args.labels) if not os.path.isabs(args.labels) else args.labels
    hidden_path = os.path.join(REPO_ROOT, args.hidden) if not os.path.isabs(args.hidden) else args.hidden

    logger.info("=" * 60)
    logger.info("EXPERIMENT 08 — STEP 3: PROBE TRAINING & ANALYSIS")
    logger.info("=" * 60)

    start_time = time.time()

    # Load data
    hidden_states, labels, domains, lengths = load_data(labels_path, hidden_path)

    # 1. Per-layer probe
    logger.info("\n--- Per-Layer Probe Accuracy ---")
    layer_results, best_layer = run_per_layer_probe(hidden_states, labels)

    # 2. Permutation test at best layer
    logger.info("\n--- Permutation Test ---")
    best_X = hidden_states[best_layer]
    perm_result = permutation_test(best_X, labels, n_permutations=500)
    logger.info(f"  Real accuracy: {perm_result['real_accuracy']:.3f}")
    logger.info(f"  p-value: {perm_result['p_value']:.4f}")

    # 3. Length baseline
    logger.info("\n--- Length Baseline ---")
    length_acc = run_length_baseline(lengths, labels)

    # 4. Per-domain analysis
    logger.info("\n--- Per-Domain Analysis ---")
    domain_results = run_per_domain_analysis(hidden_states, labels, domains, best_layer)

    # 5. Cross-domain transfer
    logger.info("\n--- Cross-Domain Transfer ---")
    transfer_results = run_cross_domain_transfer(hidden_states, labels, domains, best_layer)

    # Save all results
    all_results = {
        "experiment": "08_realworld_deception",
        "best_layer": best_layer,
        "best_accuracy": float(layer_results[best_layer]["mean"]),
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "permutation_test": perm_result,
        "length_baseline": float(length_acc),
        "per_domain": domain_results,
        "cross_domain_transfer": transfer_results,
        "n_disclosed": int(np.sum(labels == 1)),
        "n_concealed": int(np.sum(labels == 0)),
        "n_total": len(labels),
    }

    save_results(all_results, "exp08_probe_results")

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PROBE ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best layer: {best_layer}")
    logger.info(f"  Best accuracy: {layer_results[best_layer]['mean']:.3f}")
    logger.info(f"  p-value: {perm_result['p_value']:.4f}")
    logger.info(f"  Length baseline: {length_acc:.3f}")
    logger.info(f"  Total time: {elapsed:.0f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
