"""
DECEPTION PROBE - Stage 5 Part B: Classify & Train Probes
==========================================================
Loads hidden states from Part A + classifications from review.
Trains linear probes to detect deception in real-world scenarios.

CONFOUND-FREE DESIGN:
- Same prompt for all scenarios (model decides on its own)
- Classification based on whether model disclosed or hid the problem
- Labels: honest (disclosed problem) vs deceptive (hid/downplayed problem)

Usage (Colab with GPU):
    # After Part A has run and classification file is in the repo:
    %cd /content/deception-probe
    !git pull
    %run stages/stage5_realworld_deception/run_stage5_part_b.py
"""

import os
import torch
import numpy as np
import json
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("DECEPTION PROBE - Stage 5 Part B: Classify & Train Probes")
print("=" * 60)
print()
print("CONFOUND-FREE DESIGN:")
print("  - Same prompt for ALL scenarios")
print("  - Model decided on its own whether to disclose or hide")
print("  - Labels from independent review (HONEST vs DECEPTIVE/PARTIAL)")
print("=" * 60)

start_time = time.time()

# ============================================================
# STEP 1: Load Hidden States + Classifications
# ============================================================
print("\n[1/4] Loading data...")

# Find hidden states file
pkl_paths = [
    "stage5_hidden_states.pkl",
    "results/stage5_hidden_states.pkl",
    "/content/deception-probe/stage5_hidden_states.pkl",
    "/content/deception-probe/results/stage5_hidden_states.pkl",
]

pkl_path = None
for p in pkl_paths:
    if os.path.exists(p):
        pkl_path = p
        break

if pkl_path is None:
    print("  ERROR: Cannot find stage5_hidden_states.pkl")
    print("  Make sure Part A has been run first!")
    print("  Looked in:", pkl_paths)
    raise FileNotFoundError("stage5_hidden_states.pkl not found")

print(f"  Hidden states: {pkl_path}")

with open(pkl_path, "rb") as f:
    hidden_data = pickle.load(f)

print(f"  Loaded {len(hidden_data)} entries with hidden states")

# Find classification file
script_dir = os.path.dirname(os.path.abspath(__file__))
cls_paths = [
    os.path.join(script_dir, "stage5_classifications.json"),
    "stages/stage5_realworld_deception/stage5_classifications.json",
    "/content/deception-probe/stages/stage5_realworld_deception/stage5_classifications.json",
]

cls_path = None
for p in cls_paths:
    if os.path.exists(p):
        cls_path = p
        break

if cls_path is None:
    print("  ERROR: Cannot find stage5_classifications.json")
    print("  Make sure the classification file has been pushed to Git!")
    print("  Looked in:", cls_paths)
    raise FileNotFoundError("stage5_classifications.json not found")

print(f"  Classifications: {cls_path}")

with open(cls_path, "r") as f:
    classifications = json.load(f)

# Build classification lookup by index
cls_lookup = {}
for c in classifications:
    cls_lookup[c["index"]] = c["classification"]  # "honest" or "deceptive"

print(f"  Classifications loaded: {len(cls_lookup)}")
label_counts = Counter(cls_lookup.values())
print(f"    honest: {label_counts.get('honest', 0)}")
print(f"    deceptive: {label_counts.get('deceptive', 0)}")

# ============================================================
# STEP 2: Match hidden states with classifications
# ============================================================
print("\n[2/4] Matching hidden states with labels...")

matched_data = []
for entry in hidden_data:
    idx = entry["index"]
    if idx in cls_lookup:
        label = cls_lookup[idx]
        matched_data.append({
            "index": idx,
            "domain": entry["domain"],
            "response": entry["response"],
            "response_length": entry["response_length"],
            "hidden_states": entry["hidden_states"],
            "label": label,
            "label_int": 1 if label == "deceptive" else 0,
        })

print(f"  Matched: {len(matched_data)} entries")
matched_labels = Counter(m["label"] for m in matched_data)
print(f"    honest: {matched_labels.get('honest', 0)}")
print(f"    deceptive: {matched_labels.get('deceptive', 0)}")

if matched_labels.get("deceptive", 0) < 10:
    print("\n  WARNING: Very few deceptive samples. Results may not be reliable.")

# Get available layers
sample_hs = matched_data[0]["hidden_states"]
available_layers = sorted(sample_hs.keys())
print(f"  Available layers: {len(available_layers)} ({min(available_layers)}-{max(available_layers)})")

# ============================================================
# STEP 3: Train Linear Probes (per layer)
# ============================================================
print("\n[3/4] Training linear probes...")
print(f"  Method: 5-fold stratified cross-validation")
print(f"  Classifier: Logistic Regression (balanced class weights)")

labels = np.array([m["label_int"] for m in matched_data])
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

layer_results = {}

for layer_idx in available_layers:
    # Build feature matrix for this layer
    X = np.array([m["hidden_states"][layer_idx] for m in matched_data])
    y = labels

    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            C=1.0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fold_accs.append(accuracy_score(y_test, y_pred))

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    layer_results[layer_idx] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_accuracies": fold_accs,
    }

    if layer_idx % 4 == 0 or layer_idx == available_layers[-1]:
        print(f"  Layer {layer_idx:2d}: {mean_acc:.1%} (+/- {std_acc:.1%})")

# Best layer
best_layer = max(layer_results, key=lambda l: layer_results[l]["mean_accuracy"])
best_acc = layer_results[best_layer]["mean_accuracy"]
best_std = layer_results[best_layer]["std_accuracy"]

print(f"\n  BEST LAYER: {best_layer} with {best_acc:.1%} (+/- {best_std:.1%})")

# ============================================================
# STEP 4: Confound Checks
# ============================================================
print(f"\n[4/4] Running confound checks...")

# --- CHECK 1: Length Baseline ---
print(f"\n  [Check 1] Length Baseline:")
lengths = np.array([m["response_length"] for m in matched_data]).reshape(-1, 1)

length_accs = []
for train_idx, test_idx in skf.split(lengths, labels):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(lengths[train_idx], labels[train_idx])
    y_pred = clf.predict(lengths[test_idx])
    length_accs.append(accuracy_score(labels[test_idx], y_pred))

length_baseline = np.mean(length_accs)
print(f"    Length-only accuracy: {length_baseline:.1%}")
if length_baseline > 0.65:
    print(f"    WARNING: Length baseline is high — response length may be a confound")
else:
    print(f"    OK: Length alone is not a strong predictor")

# --- CHECK 2: Permutation Test ---
print(f"\n  [Check 2] Permutation Test (500 iterations):")
X_best = np.array([m["hidden_states"][best_layer] for m in matched_data])

perm_accs = []
for _ in range(500):
    shuffled_labels = np.random.permutation(labels)
    fold_accs_perm = []
    for train_idx, test_idx in skf.split(X_best, shuffled_labels):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_best[train_idx], shuffled_labels[train_idx])
        y_pred = clf.predict(X_best[test_idx])
        fold_accs_perm.append(accuracy_score(shuffled_labels[test_idx], y_pred))
    perm_accs.append(np.mean(fold_accs_perm))

p_value = np.mean([pa >= best_acc for pa in perm_accs])
print(f"    Real accuracy: {best_acc:.1%}")
print(f"    Permutation mean: {np.mean(perm_accs):.1%} (+/- {np.std(perm_accs):.1%})")
print(f"    p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"    SIGNIFICANT (p < 0.05)")
else:
    print(f"    NOT SIGNIFICANT (p >= 0.05)")

# --- CHECK 3: Cross-Domain Generalization ---
print(f"\n  [Check 3] Cross-Domain Generalization (leave-one-domain-out):")

# Get top-level domains
domains = list(set(m["domain"].split(" - ")[0] for m in matched_data))
domain_results = {}

for test_domain in sorted(domains):
    train_mask = [m["domain"].split(" - ")[0] != test_domain for m in matched_data]
    test_mask = [m["domain"].split(" - ")[0] == test_domain for m in matched_data]

    X_train = X_best[train_mask]
    y_train = labels[train_mask]
    X_test = X_best[test_mask]
    y_test = labels[test_mask]

    # Skip if test set has only one class
    if len(set(y_test)) < 2:
        domain_results[test_domain] = {"accuracy": None, "n_test": sum(test_mask), "note": "single class"}
        continue

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    domain_results[test_domain] = {"accuracy": acc, "n_test": sum(test_mask)}

print(f"    {'Domain':<20} {'Accuracy':>10} {'N_test':>8}")
print(f"    {'-'*40}")
valid_domain_accs = []
for domain in sorted(domain_results.keys()):
    r = domain_results[domain]
    if r["accuracy"] is not None:
        print(f"    {domain:<20} {r['accuracy']:>10.1%} {r['n_test']:>8}")
        valid_domain_accs.append(r["accuracy"])
    else:
        print(f"    {domain:<20} {'N/A':>10} {r['n_test']:>8}  ({r['note']})")

if valid_domain_accs:
    print(f"\n    Mean cross-domain accuracy: {np.mean(valid_domain_accs):.1%}")

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

results = {
    "stage": "Stage 5 - Real-World Deception (Confound-Free)",
    "design": {
        "prompt": "IDENTICAL for all scenarios",
        "model_behavior": "Model decides on its own whether to disclose or hide problems",
        "classification": "Independent review (HONEST vs DECEPTIVE/PARTIAL)",
        "confound_free": True,
    },
    "data": {
        "total_scenarios": len(matched_data),
        "honest": int(matched_labels.get("honest", 0)),
        "deceptive": int(matched_labels.get("deceptive", 0)),
        "deception_rate": float(matched_labels.get("deceptive", 0)) / len(matched_data),
    },
    "probe_results": {
        "best_layer": int(best_layer),
        "best_accuracy": float(best_acc),
        "best_std": float(best_std),
        "all_layers": {
            int(k): {"mean": float(v["mean_accuracy"]), "std": float(v["std_accuracy"])}
            for k, v in layer_results.items()
        },
    },
    "confound_checks": {
        "length_baseline": float(length_baseline),
        "permutation_p_value": float(p_value),
        "cross_domain_mean": float(np.mean(valid_domain_accs)) if valid_domain_accs else None,
        "cross_domain_results": {
            k: {"accuracy": float(v["accuracy"]) if v["accuracy"] is not None else None, "n_test": v["n_test"]}
            for k, v in domain_results.items()
        },
    },
}

with open("stage5_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - start_time

print(f"\n{'='*60}")
print("STAGE 5 RESULTS SUMMARY")
print(f"{'='*60}")
print(f"\n  Design: CONFOUND-FREE (same prompt, model decides)")
print(f"  Data: {len(matched_data)} scenarios ({matched_labels.get('honest',0)} honest, {matched_labels.get('deceptive',0)} deceptive)")
print(f"  Deception rate: {matched_labels.get('deceptive',0)/len(matched_data)*100:.1f}%")
print(f"\n  PROBE ACCURACY (best layer {best_layer}): {best_acc:.1%} (+/- {best_std:.1%})")
print(f"\n  Confound Checks:")
print(f"    Length baseline:    {length_baseline:.1%} {'(WARNING)' if length_baseline > 0.65 else '(OK)'}")
print(f"    Permutation p-val:  {p_value:.4f} {'(SIGNIFICANT)' if p_value < 0.05 else '(NOT SIGNIFICANT)'}")
if valid_domain_accs:
    print(f"    Cross-domain mean:  {np.mean(valid_domain_accs):.1%}")

print(f"\n  Layer Profile:")
for layer_idx in available_layers:
    acc = layer_results[layer_idx]["mean_accuracy"]
    bar = "#" * int(acc * 40)
    marker = " <-- BEST" if layer_idx == best_layer else ""
    print(f"    Layer {layer_idx:2d}: {acc:.1%} |{bar}{marker}")

print(f"\n  Comparison with Previous Stages:")
print(f"    Stage 1-3 (confounded):     100% (prompt confound)")
print(f"    Stage 4 (trivia, clean):    82.5%")
print(f"    Stage 5 (real-world, clean): {best_acc:.1%}")

print(f"\n  Interpretation:")
if best_acc > 0.80 and p_value < 0.05 and length_baseline < 0.65:
    print(f"    STRONG SIGNAL: Probe detects real-world deception!")
    print(f"    The signal is not explained by length or random chance.")
elif best_acc > 0.65 and p_value < 0.05:
    print(f"    MODERATE SIGNAL: Probe detects deception above chance.")
    print(f"    Signal is statistically significant but may have confounds.")
elif best_acc > 0.55 and p_value < 0.05:
    print(f"    WEAK SIGNAL: Small but significant deception signal detected.")
else:
    print(f"    NO CLEAR SIGNAL: Probe cannot reliably detect real-world deception.")
    print(f"    This suggests the model's internal state doesn't differ much")
    print(f"    between honest and deceptive responses in real-world scenarios.")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"\n  Saved: stage5_results.json")
print(f"{'='*60}")
print("STAGE 5 COMPLETE")
print(f"{'='*60}")
