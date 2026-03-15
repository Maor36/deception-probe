"""
DECEPTION PROBE - Stage 5 Part B: Classify & Train Probes (v2)
================================================================
Loads hidden states from Part A + classifications from review.
Trains probes to detect deception in real-world scenarios.

FIXES in v2:
- Skip layer 0 (embedding) for best-layer selection
- Permutation test on real best layer (not embedding)
- Learning curve: does more data help?
- Multiple classifiers: LogReg, SVM, Random Forest, MLP

Usage (Colab with GPU):
    %cd /content/deception-probe
    !git pull
    %run stages/stage5_realworld_deception/run_stage5_part_b.py
"""

import os
import numpy as np
import json
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
SKIP_LAYER_0 = True  # Layer 0 = embedding, not meaningful for deception detection

print("=" * 60)
print("DECEPTION PROBE - Stage 5 Part B v2")
print("=" * 60)
print()
print("CONFOUND-FREE DESIGN:")
print("  - Same prompt for ALL scenarios")
print("  - Model decided on its own whether to disclose or hide")
print("  - Labels from independent review (HONEST vs DECEPTIVE/PARTIAL)")
print("  - Layer 0 (embedding) EXCLUDED from best-layer selection")
print("=" * 60)

start_time = time.time()

# ============================================================
# STEP 1: Load Hidden States + Classifications
# ============================================================
print("\n[1/6] Loading data...")

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
    raise FileNotFoundError("stage5_hidden_states.pkl not found. Run Part A first!")

print(f"  Hidden states: {pkl_path}")
with open(pkl_path, "rb") as f:
    hidden_data = pickle.load(f)
print(f"  Loaded {len(hidden_data)} entries")

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
    raise FileNotFoundError("stage5_classifications.json not found. Push classifications first!")

print(f"  Classifications: {cls_path}")
with open(cls_path, "r") as f:
    classifications = json.load(f)

cls_lookup = {c["index"]: c["classification"] for c in classifications}
label_counts = Counter(cls_lookup.values())
print(f"  Labels: {label_counts}")

# ============================================================
# STEP 2: Match hidden states with classifications
# ============================================================
print("\n[2/6] Matching data...")

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
print(f"    deception rate: {matched_labels.get('deceptive', 0)/len(matched_data)*100:.1f}%")

sample_hs = matched_data[0]["hidden_states"]
available_layers = sorted(sample_hs.keys())
probe_layers = [l for l in available_layers if l > 0] if SKIP_LAYER_0 else available_layers
print(f"  Probe layers: {len(probe_layers)} (skipping layer 0: {SKIP_LAYER_0})")

labels = np.array([m["label_int"] for m in matched_data])
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

# ============================================================
# STEP 3: Train Logistic Regression Probes (per layer)
# ============================================================
print("\n[3/6] Training LogReg probes per layer...")
print(f"  Method: {n_folds}-fold stratified CV, balanced class weights")

layer_results = {}

for layer_idx in available_layers:
    X = np.array([m["hidden_states"][layer_idx] for m in matched_data])
    y = labels

    fold_accs = []
    fold_bal_accs = []
    fold_f1s = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        fold_accs.append(accuracy_score(y[test_idx], y_pred))
        fold_bal_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
        fold_f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))

    layer_results[layer_idx] = {
        "mean_accuracy": np.mean(fold_accs),
        "std_accuracy": np.std(fold_accs),
        "mean_balanced_accuracy": np.mean(fold_bal_accs),
        "mean_f1": np.mean(fold_f1s),
    }

    if layer_idx % 4 == 0 or layer_idx == available_layers[-1]:
        skip_note = " (EMBEDDING - excluded)" if layer_idx == 0 else ""
        print(f"  Layer {layer_idx:2d}: acc={np.mean(fold_accs):.1%}  bal_acc={np.mean(fold_bal_accs):.1%}  f1={np.mean(fold_f1s):.3f}{skip_note}")

# Best layer (excluding layer 0)
best_layer = max(probe_layers, key=lambda l: layer_results[l]["mean_balanced_accuracy"])
best_acc = layer_results[best_layer]["mean_accuracy"]
best_bal_acc = layer_results[best_layer]["mean_balanced_accuracy"]
best_f1 = layer_results[best_layer]["mean_f1"]
best_std = layer_results[best_layer]["std_accuracy"]

print(f"\n  BEST LAYER (excl. embedding): {best_layer}")
print(f"    Accuracy:          {best_acc:.1%} (+/- {best_std:.1%})")
print(f"    Balanced Accuracy: {best_bal_acc:.1%}")
print(f"    F1 (deceptive):    {best_f1:.3f}")

# Also show layer 0 for comparison
l0_acc = layer_results[0]["mean_accuracy"]
l0_bal = layer_results[0]["mean_balanced_accuracy"]
majority_baseline = max(matched_labels.values()) / len(matched_data)
print(f"\n  Layer 0 (embedding): acc={l0_acc:.1%}, bal_acc={l0_bal:.1%}")
print(f"  Majority baseline:   acc={majority_baseline:.1%}")
print(f"  NOTE: Layer 0 acc ({l0_acc:.1%}) ~ majority baseline ({majority_baseline:.1%}) = NOT a real signal")

# ============================================================
# STEP 4: Multiple Classifiers on Best Layer
# ============================================================
print(f"\n[4/6] Comparing classifiers on layer {best_layer}...")

X_best = np.array([m["hidden_states"][best_layer] for m in matched_data])

classifiers = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    "SVM-RBF": SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED),
    "SVM-Linear": SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=RANDOM_SEED, early_stopping=True),
}

clf_results = {}
print(f"\n  {'Classifier':<20} {'Accuracy':>10} {'Bal.Acc':>10} {'F1':>10}")
print(f"  {'-'*52}")

for name, clf in classifiers.items():
    fold_accs = []
    fold_bal_accs = []
    fold_f1s = []

    for train_idx, test_idx in skf.split(X_best, labels):
        X_train, X_test = X_best[train_idx], X_best[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Scale for SVM and MLP
        if name in ("SVM-RBF", "SVM-Linear", "MLP"):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(X_train, y_train)
        y_pred = clf_copy.predict(X_test)

        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

    clf_results[name] = {
        "accuracy": np.mean(fold_accs),
        "balanced_accuracy": np.mean(fold_bal_accs),
        "f1": np.mean(fold_f1s),
    }
    print(f"  {name:<20} {np.mean(fold_accs):>10.1%} {np.mean(fold_bal_accs):>10.1%} {np.mean(fold_f1s):>10.3f}")

best_clf_name = max(clf_results, key=lambda c: clf_results[c]["balanced_accuracy"])
print(f"\n  Best classifier: {best_clf_name} (bal.acc={clf_results[best_clf_name]['balanced_accuracy']:.1%})")

# ============================================================
# STEP 5: Confound Checks (on real best layer)
# ============================================================
print(f"\n[5/6] Confound checks (on layer {best_layer})...")

# --- CHECK 1: Length Baseline ---
print(f"\n  [Check 1] Length Baseline:")
lengths = np.array([m["response_length"] for m in matched_data]).reshape(-1, 1)

length_accs = []
for train_idx, test_idx in skf.split(lengths, labels):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(lengths[train_idx], labels[train_idx])
    y_pred = clf.predict(lengths[test_idx])
    length_accs.append(balanced_accuracy_score(labels[test_idx], y_pred))

length_baseline = np.mean(length_accs)
print(f"    Length-only balanced accuracy: {length_baseline:.1%}")
if length_baseline > 0.60:
    print(f"    WARNING: Length may be a confound")
else:
    print(f"    OK: Length alone is not a strong predictor")

# --- CHECK 2: Permutation Test (on real best layer, NOT layer 0) ---
print(f"\n  [Check 2] Permutation Test (500 iterations, layer {best_layer}):")

perm_bal_accs = []
for i in range(500):
    shuffled_labels = np.random.permutation(labels)
    fold_bal_accs_perm = []
    for train_idx, test_idx in skf.split(X_best, shuffled_labels):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_best[train_idx], shuffled_labels[train_idx])
        y_pred = clf.predict(X_best[test_idx])
        fold_bal_accs_perm.append(balanced_accuracy_score(shuffled_labels[test_idx], y_pred))
    perm_bal_accs.append(np.mean(fold_bal_accs_perm))
    if (i + 1) % 100 == 0:
        print(f"    ... {i+1}/500 permutations done")

p_value = np.mean([pa >= best_bal_acc for pa in perm_bal_accs])
print(f"    Real balanced accuracy: {best_bal_acc:.1%}")
print(f"    Permutation mean: {np.mean(perm_bal_accs):.1%} (+/- {np.std(perm_bal_accs):.1%})")
print(f"    p-value: {p_value:.4f}")
if p_value < 0.001:
    print(f"    HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value < 0.05:
    print(f"    SIGNIFICANT (p < 0.05)")
else:
    print(f"    NOT SIGNIFICANT (p >= 0.05)")

# --- CHECK 3: Cross-Domain Generalization ---
print(f"\n  [Check 3] Cross-Domain Generalization (leave-one-domain-out):")

domains = list(set(m["domain"].split(" - ")[0] for m in matched_data))
domain_results = {}

for test_domain in sorted(domains):
    train_mask = np.array([m["domain"].split(" - ")[0] != test_domain for m in matched_data])
    test_mask = np.array([m["domain"].split(" - ")[0] == test_domain for m in matched_data])

    y_test = labels[test_mask]

    if len(set(y_test)) < 2:
        domain_results[test_domain] = {"bal_accuracy": None, "n_test": int(test_mask.sum()), "note": "single class"}
        continue

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_best[train_mask], labels[train_mask])
    y_pred = clf.predict(X_best[test_mask])
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    domain_results[test_domain] = {"bal_accuracy": bal_acc, "n_test": int(test_mask.sum())}

print(f"    {'Domain':<20} {'Bal.Acc':>10} {'N_test':>8}")
print(f"    {'-'*40}")
valid_domain_accs = []
for domain in sorted(domain_results.keys()):
    r = domain_results[domain]
    if r["bal_accuracy"] is not None:
        print(f"    {domain:<20} {r['bal_accuracy']:>10.1%} {r['n_test']:>8}")
        valid_domain_accs.append(r["bal_accuracy"])
    else:
        print(f"    {domain:<20} {'N/A':>10} {r['n_test']:>8}  ({r['note']})")

if valid_domain_accs:
    print(f"\n    Mean cross-domain balanced accuracy: {np.mean(valid_domain_accs):.1%}")

# ============================================================
# STEP 6: Learning Curve (does more data help?)
# ============================================================
print(f"\n[6/6] Learning curve analysis (layer {best_layer})...")

train_sizes_abs = [20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400]
# Filter to sizes that are feasible
max_train = int(len(matched_data) * 0.8)  # 80% for training in 5-fold
train_sizes_abs = [s for s in train_sizes_abs if s < max_train]

try:
    train_sizes_out, train_scores, test_scores = learning_curve(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        X_best, labels,
        train_sizes=train_sizes_abs,
        cv=skf,
        scoring="balanced_accuracy",
        random_state=RANDOM_SEED,
    )

    print(f"\n  {'Train Size':>12} {'Train Bal.Acc':>14} {'Test Bal.Acc':>14} {'Gap':>8}")
    print(f"  {'-'*50}")
    for i, size in enumerate(train_sizes_out):
        train_mean = np.mean(train_scores[i])
        test_mean = np.mean(test_scores[i])
        gap = train_mean - test_mean
        print(f"  {size:>12} {train_mean:>14.1%} {test_mean:>14.1%} {gap:>8.1%}")

    # Extrapolation hint
    last_test = np.mean(test_scores[-1])
    second_last_test = np.mean(test_scores[-2]) if len(test_scores) > 1 else last_test
    improvement = last_test - second_last_test
    print(f"\n  Last improvement: {improvement:+.1%}")
    if improvement > 0.01:
        print(f"  TREND: Still improving — more data would likely help!")
    elif improvement > 0:
        print(f"  TREND: Marginal improvement — approaching plateau")
    else:
        print(f"  TREND: Plateaued — more data unlikely to help significantly")
except Exception as e:
    print(f"  Learning curve failed: {e}")

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

results = {
    "stage": "Stage 5 - Real-World Deception (Confound-Free) v2",
    "design": {
        "prompt": "IDENTICAL for all scenarios",
        "model_behavior": "Model decides on its own whether to disclose or hide",
        "classification": "Independent review (HONEST vs DECEPTIVE/PARTIAL)",
        "confound_free": True,
        "layer_0_excluded": True,
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
        "best_balanced_accuracy": float(best_bal_acc),
        "best_f1": float(best_f1),
        "all_layers": {
            int(k): {
                "accuracy": float(v["mean_accuracy"]),
                "balanced_accuracy": float(v["mean_balanced_accuracy"]),
                "f1": float(v["mean_f1"]),
            }
            for k, v in layer_results.items()
        },
    },
    "classifier_comparison": {
        name: {k: float(v) for k, v in res.items()}
        for name, res in clf_results.items()
    },
    "confound_checks": {
        "length_baseline_bal_acc": float(length_baseline),
        "permutation_p_value": float(p_value),
        "permutation_layer": int(best_layer),
        "cross_domain_mean_bal_acc": float(np.mean(valid_domain_accs)) if valid_domain_accs else None,
    },
}

with open("stage5_results_v2.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - start_time

print(f"\n{'='*60}")
print("STAGE 5 RESULTS SUMMARY (v2)")
print(f"{'='*60}")
print(f"\n  Design: CONFOUND-FREE (same prompt, model decides)")
print(f"  Data: {len(matched_data)} scenarios ({matched_labels.get('honest',0)} honest, {matched_labels.get('deceptive',0)} deceptive)")
print(f"  Deception rate: {matched_labels.get('deceptive',0)/len(matched_data)*100:.1f}%")

print(f"\n  PROBE (best layer {best_layer}, excluding embedding):")
print(f"    Accuracy:          {best_acc:.1%}")
print(f"    Balanced Accuracy: {best_bal_acc:.1%}")
print(f"    F1 (deceptive):    {best_f1:.3f}")

print(f"\n  CLASSIFIER COMPARISON (layer {best_layer}):")
for name in sorted(clf_results, key=lambda c: clf_results[c]["balanced_accuracy"], reverse=True):
    r = clf_results[name]
    print(f"    {name:<20} bal_acc={r['balanced_accuracy']:.1%}  f1={r['f1']:.3f}")

print(f"\n  CONFOUND CHECKS:")
print(f"    Length baseline:    {length_baseline:.1%} {'(WARNING)' if length_baseline > 0.60 else '(OK)'}")
print(f"    Permutation p-val:  {p_value:.4f} {'(SIGNIFICANT)' if p_value < 0.05 else '(NOT SIGNIFICANT)'}")
if valid_domain_accs:
    print(f"    Cross-domain mean:  {np.mean(valid_domain_accs):.1%}")

print(f"\n  LAYER PROFILE (balanced accuracy):")
for layer_idx in available_layers:
    bal = layer_results[layer_idx]["mean_balanced_accuracy"]
    bar = "#" * int(bal * 40)
    marker = " <-- BEST" if layer_idx == best_layer else ""
    skip = " (embedding)" if layer_idx == 0 else ""
    print(f"    Layer {layer_idx:2d}: {bal:.1%} |{bar}{marker}{skip}")

print(f"\n  COMPARISON WITH PREVIOUS STAGES:")
print(f"    Stage 1-3 (confounded):      100% (prompt confound)")
print(f"    Stage 4 (trivia, clean):     82.5% accuracy")
print(f"    Stage 5 (real-world, clean): {best_bal_acc:.1%} balanced accuracy (layer {best_layer})")

print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Saved: stage5_results_v2.json")
print(f"{'='*60}")
print("STAGE 5 COMPLETE")
print(f"{'='*60}")
