"""
DECEPTION PROBE - Stage 7: Advanced Hallucination Detection
=============================================================
Goal: Can we reliably detect when a model is HALLUCINATING using
      multiple complementary methods on hidden states?

Stage 6 showed:
  - Truth vs Lie:           100% (easy - model "knows" it's lying)
  - Lie vs Hallucination:   100% (easy - different internal states)
  - Truth vs Hallucination:  67% (hard - no "tension" signal)

This stage tries MULTIPLE methods to improve hallucination detection:

METHOD 1: Multi-layer fusion
  Instead of one layer, combine features from multiple layers.
  Intuition: hallucination signal may be distributed across layers.

METHOD 2: Layer difference vectors
  Use the DIFFERENCE between early and late layer representations.
  Intuition: when the model hallucinates, information flow between
  layers may differ from truthful responses.

METHOD 3: Layer statistics (variance, norm, entropy-like features)
  Compute statistical properties of hidden states across layers.
  Intuition: hallucinating models may show different activation patterns.

METHOD 4: Contrastive pairs
  Train on (Truth, Hallucination) pairs from SAME topic/domain.
  Intuition: controlling for topic removes content-based confounds.

METHOD 5: Ensemble of all methods
  Combine predictions from all methods.

METHOD 6: PCA direction discovery
  Find the principal direction that separates Truth from Hallucination.
  Like "truth direction" papers but for hallucination.

This script reuses hidden states saved from Stage 6 (no GPU needed!).

Usage (Colab, no GPU required):
    !pip install -q scikit-learn
    %cd /content/deception-probe
    %run stages/stage7_hallucination_detection/run_stage7.py
"""

import os
import numpy as np
import json
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, make_scorer, confusion_matrix
from sklearn.pipeline import Pipeline

np.random.seed(42)

print("=" * 60)
print("DECEPTION PROBE - Stage 7: Advanced Hallucination Detection")
print("=" * 60)

start_time = time.time()

# ============================================================
# STEP 1: Load cached hidden states from Stage 6
# ============================================================
print("\n[1/8] Loading cached hidden states from Stage 6...")

hs_path = "results/stage6_hidden_states.pkl"
if not os.path.exists(hs_path):
    print(f"  ERROR: {hs_path} not found!")
    print(f"  Run Stage 6 first to generate hidden states.")
    exit(1)

with open(hs_path, "rb") as f:
    data = pickle.load(f)

truth_data = data["truth"]
lie_data = data["lie"]
hall_data = data["hallucination"]
probe_layers = data["probe_layers"]
model_name = data["model"]

print(f"  Model: {model_name}")
print(f"  Layers available: {probe_layers}")
print(f"  TRUTH:         {len(truth_data)}")
print(f"  LIE:           {len(lie_data)}")
print(f"  HALLUCINATION: {len(hall_data)}")

# Balance classes
min_n = min(len(truth_data), len(hall_data))
print(f"  Balanced (Truth vs Hall): {min_n} per class")

bal_acc_scorer = make_scorer(balanced_accuracy_score)

# Skip layer 0 (embedding)
layers_no_emb = [l for l in probe_layers if l != 0]

# ============================================================
# Helper: get features for a specific layer
# ============================================================
def get_layer_features(dataset, layer_idx, n=None):
    if n is None:
        n = len(dataset)
    return np.array([d["layer_hs"][layer_idx] for d in dataset[:n]])


def evaluate_method(X, y, method_name, n_splits=5):
    """Evaluate a method with cross-validation and return results."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y))),
                         shuffle=True, random_state=42)
    
    classifiers = {
        "LogReg": LogisticRegression(max_iter=2000, random_state=42, C=1.0),
        "SVM-RBF": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        "GradBoost": GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                  random_state=42),
    }
    
    best_name = None
    best_score = 0
    all_scores = {}
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring=bal_acc_scorer)
        all_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
    
    return {
        "method": method_name,
        "best_classifier": best_name,
        "best_bal_acc": float(best_score),
        "all_classifiers": all_scores,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
    }


# ============================================================
# BASELINE: Single best layer (from Stage 6)
# ============================================================
print("\n[2/8] Baseline: Single layer (reproducing Stage 6)...")

baseline_results = {}
for layer_idx in layers_no_emb:
    X_t = get_layer_features(truth_data, layer_idx, min_n)
    X_h = get_layer_features(hall_data, layer_idx, min_n)
    X = np.vstack([X_t, X_h])
    y = np.array([0] * min_n + [1] * min_n)
    
    X_s = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_s, y, cv=cv, scoring=bal_acc_scorer)
    baseline_results[layer_idx] = float(scores.mean())
    print(f"  Layer {layer_idx:2d}: {scores.mean()*100:.1f}%")

best_single_layer = max(baseline_results, key=baseline_results.get)
best_single_acc = baseline_results[best_single_layer]
print(f"  BEST single layer: {best_single_layer} ({best_single_acc*100:.1f}%)")

results_all = {}
results_all["baseline_single_layer"] = {
    "layer": best_single_layer,
    "bal_acc": best_single_acc,
}

# ============================================================
# METHOD 1: Multi-layer fusion
# ============================================================
print("\n[3/8] Method 1: Multi-layer fusion...")

# Try different layer combinations
layer_combos = {
    "top5": sorted(baseline_results, key=baseline_results.get, reverse=True)[:5],
    "top3": sorted(baseline_results, key=baseline_results.get, reverse=True)[:3],
    "middle_layers": [l for l in layers_no_emb if 14 <= l <= 22],
    "all_layers": layers_no_emb,
    "early+late": [l for l in layers_no_emb if l <= 6 or l >= 26],
    "every_4th": [l for l in layers_no_emb if l % 4 == 0],
}

print(f"  {'Combination':<20s} {'Layers':>8s} {'Features':>10s} {'Bal.Acc':>10s}")
print(f"  {'-'*55}")

best_fusion_name = None
best_fusion_acc = 0

for combo_name, layers in layer_combos.items():
    X_t_parts = [get_layer_features(truth_data, l, min_n) for l in layers]
    X_h_parts = [get_layer_features(hall_data, l, min_n) for l in layers]
    
    X_t = np.hstack(X_t_parts)
    X_h = np.hstack(X_h_parts)
    X = np.vstack([X_t, X_h])
    y = np.array([0] * min_n + [1] * min_n)
    
    result = evaluate_method(X, y, f"fusion_{combo_name}")
    ba = result["best_bal_acc"]
    
    print(f"  {combo_name:<20s} {len(layers):>8d} {X.shape[1]:>10d} {ba*100:>9.1f}%")
    
    if ba > best_fusion_acc:
        best_fusion_acc = ba
        best_fusion_name = combo_name

results_all["method1_fusion"] = {
    "best_combo": best_fusion_name,
    "bal_acc": best_fusion_acc,
}
print(f"  BEST fusion: {best_fusion_name} ({best_fusion_acc*100:.1f}%)")

# ============================================================
# METHOD 2: Layer difference vectors
# ============================================================
print("\n[4/8] Method 2: Layer difference vectors...")

# Compute differences between layer pairs
diff_pairs = [
    ("early_vs_late", [2, 4, 6], [26, 28, 30]),
    ("early_vs_mid", [2, 4, 6], [14, 16, 18]),
    ("mid_vs_late", [14, 16, 18], [26, 28, 30]),
    ("first_vs_best", [2], [best_single_layer]),
    ("gradient_all", None, None),  # special: consecutive diffs
]

print(f"  {'Pair':<20s} {'Features':>10s} {'Bal.Acc':>10s}")
print(f"  {'-'*45}")

best_diff_name = None
best_diff_acc = 0

for pair_name, early_layers, late_layers in diff_pairs:
    if pair_name == "gradient_all":
        # Consecutive layer differences
        sorted_layers = sorted(layers_no_emb)
        X_t_diffs = []
        X_h_diffs = []
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            t1 = get_layer_features(truth_data, l1, min_n)
            t2 = get_layer_features(truth_data, l2, min_n)
            h1 = get_layer_features(hall_data, l1, min_n)
            h2 = get_layer_features(hall_data, l2, min_n)
            X_t_diffs.append(t2 - t1)
            X_h_diffs.append(h2 - h1)
        X_t_feat = np.hstack(X_t_diffs)
        X_h_feat = np.hstack(X_h_diffs)
    else:
        # Average early layers, average late layers, take diff
        early_layers_valid = [l for l in early_layers if l in layers_no_emb]
        late_layers_valid = [l for l in late_layers if l in layers_no_emb]
        
        if not early_layers_valid or not late_layers_valid:
            continue
        
        X_t_early = np.mean([get_layer_features(truth_data, l, min_n) for l in early_layers_valid], axis=0)
        X_t_late = np.mean([get_layer_features(truth_data, l, min_n) for l in late_layers_valid], axis=0)
        X_h_early = np.mean([get_layer_features(hall_data, l, min_n) for l in early_layers_valid], axis=0)
        X_h_late = np.mean([get_layer_features(hall_data, l, min_n) for l in late_layers_valid], axis=0)
        
        X_t_feat = X_t_late - X_t_early
        X_h_feat = X_h_late - X_h_early
    
    X = np.vstack([X_t_feat, X_h_feat])
    y = np.array([0] * min_n + [1] * min_n)
    
    result = evaluate_method(X, y, f"diff_{pair_name}")
    ba = result["best_bal_acc"]
    
    print(f"  {pair_name:<20s} {X.shape[1]:>10d} {ba*100:>9.1f}%")
    
    if ba > best_diff_acc:
        best_diff_acc = ba
        best_diff_name = pair_name

results_all["method2_layer_diff"] = {
    "best_pair": best_diff_name,
    "bal_acc": best_diff_acc,
}
print(f"  BEST diff: {best_diff_name} ({best_diff_acc*100:.1f}%)")

# ============================================================
# METHOD 3: Layer statistics (meta-features)
# ============================================================
print("\n[5/8] Method 3: Layer statistics (meta-features)...")

def compute_layer_stats(dataset, n=None):
    """Compute statistical features across all layers for each sample."""
    if n is None:
        n = len(dataset)
    
    features = []
    for i in range(n):
        sample_features = []
        
        layer_vecs = []
        for l in layers_no_emb:
            vec = dataset[i]["layer_hs"][l]
            layer_vecs.append(vec)
        
        layer_vecs = np.array(layer_vecs)  # (n_layers, hidden_dim)
        
        # Per-layer statistics
        norms = np.linalg.norm(layer_vecs, axis=1)  # norm of each layer
        means = np.mean(layer_vecs, axis=1)  # mean activation per layer
        stds = np.std(layer_vecs, axis=1)  # std per layer
        maxes = np.max(np.abs(layer_vecs), axis=1)  # max abs activation
        
        # Cross-layer statistics
        sample_features.extend(norms)           # norm per layer
        sample_features.extend(stds)            # std per layer
        sample_features.extend(maxes)           # max abs per layer
        
        # Norm trajectory features
        sample_features.append(np.mean(norms))      # avg norm
        sample_features.append(np.std(norms))       # norm variability
        sample_features.append(norms[-1] - norms[0])  # norm change
        sample_features.append(np.max(norms) - np.min(norms))  # norm range
        
        # Consecutive layer cosine similarities
        for j in range(len(layer_vecs) - 1):
            cos_sim = np.dot(layer_vecs[j], layer_vecs[j+1]) / (
                np.linalg.norm(layer_vecs[j]) * np.linalg.norm(layer_vecs[j+1]) + 1e-8)
            sample_features.append(cos_sim)
        
        # Entropy-like: how "spread out" are activations
        for l_idx in range(len(layer_vecs)):
            abs_vec = np.abs(layer_vecs[l_idx])
            prob = abs_vec / (abs_vec.sum() + 1e-8)
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            sample_features.append(entropy)
        
        # Sparsity: fraction of near-zero activations
        for l_idx in range(len(layer_vecs)):
            sparsity = np.mean(np.abs(layer_vecs[l_idx]) < 0.01)
            sample_features.append(sparsity)
        
        features.append(sample_features)
    
    return np.array(features)

X_t_stats = compute_layer_stats(truth_data, min_n)
X_h_stats = compute_layer_stats(hall_data, min_n)
X_stats = np.vstack([X_t_stats, X_h_stats])
y_stats = np.array([0] * min_n + [1] * min_n)

result_stats = evaluate_method(X_stats, y_stats, "layer_statistics")
print(f"  Features: {X_stats.shape[1]}")
for name, scores in result_stats["all_classifiers"].items():
    print(f"  {name:<15s}: {scores['mean']*100:.1f}% +/- {scores['std']*100:.1f}%")

results_all["method3_layer_stats"] = {
    "bal_acc": result_stats["best_bal_acc"],
    "best_classifier": result_stats["best_classifier"],
    "n_features": int(X_stats.shape[1]),
}
print(f"  BEST: {result_stats['best_classifier']} ({result_stats['best_bal_acc']*100:.1f}%)")

# ============================================================
# METHOD 4: Combined features (best layer + stats + diffs)
# ============================================================
print("\n[6/8] Method 4: Combined features (fusion + stats + diffs)...")

# Best single layer features
X_t_best = get_layer_features(truth_data, best_single_layer, min_n)
X_h_best = get_layer_features(hall_data, best_single_layer, min_n)

# Best diff features
sorted_layers = sorted(layers_no_emb)
X_t_diffs = []
X_h_diffs = []
for i in range(len(sorted_layers) - 1):
    l1, l2 = sorted_layers[i], sorted_layers[i + 1]
    t1 = get_layer_features(truth_data, l1, min_n)
    t2 = get_layer_features(truth_data, l2, min_n)
    h1 = get_layer_features(hall_data, l1, min_n)
    h2 = get_layer_features(hall_data, l2, min_n)
    X_t_diffs.append(t2 - t1)
    X_h_diffs.append(h2 - h1)
X_t_diff_all = np.hstack(X_t_diffs)
X_h_diff_all = np.hstack(X_h_diffs)

# Combine: best layer + stats + diffs
X_t_combined = np.hstack([X_t_best, X_t_stats, X_t_diff_all])
X_h_combined = np.hstack([X_h_best, X_h_stats, X_h_diff_all])
X_combined = np.vstack([X_t_combined, X_h_combined])
y_combined = np.array([0] * min_n + [1] * min_n)

# Use PCA to reduce dimensionality before classification
pca_dims = [50, 100, 200, 500]
print(f"  Total raw features: {X_combined.shape[1]}")
print(f"  {'PCA dims':<12s} {'Bal.Acc':>10s}")
print(f"  {'-'*25}")

best_pca_dim = None
best_pca_acc = 0

for n_comp in pca_dims:
    if n_comp >= X_combined.shape[0]:
        continue
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_combined, y_combined, cv=cv, scoring=bal_acc_scorer)
    
    print(f"  {n_comp:<12d} {scores.mean()*100:>9.1f}%")
    
    if scores.mean() > best_pca_acc:
        best_pca_acc = scores.mean()
        best_pca_dim = n_comp

# Also try without PCA but with regularized classifiers
result_combined = evaluate_method(X_combined, y_combined, "combined_all")
no_pca_acc = result_combined["best_bal_acc"]
print(f"  No PCA     {no_pca_acc*100:>9.1f}% ({result_combined['best_classifier']})")

final_combined_acc = max(best_pca_acc, no_pca_acc)
results_all["method4_combined"] = {
    "bal_acc": final_combined_acc,
    "best_pca_dim": best_pca_dim if best_pca_acc > no_pca_acc else None,
    "n_raw_features": int(X_combined.shape[1]),
}
print(f"  BEST combined: {final_combined_acc*100:.1f}%")

# ============================================================
# METHOD 5: PCA direction discovery ("hallucination direction")
# ============================================================
print("\n[7/8] Method 5: Hallucination direction (PCA)...")

print(f"  Looking for a 'hallucination direction' in hidden state space...")

best_pca_layer = None
best_pca_layer_acc = 0

for layer_idx in layers_no_emb:
    X_t = get_layer_features(truth_data, layer_idx, min_n)
    X_h = get_layer_features(hall_data, layer_idx, min_n)
    
    # Compute mean difference vector (candidate "hallucination direction")
    mean_diff = X_h.mean(axis=0) - X_t.mean(axis=0)
    mean_diff = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    
    # Project all samples onto this direction
    X_all = np.vstack([X_t, X_h])
    projections = X_all @ mean_diff
    
    # Use projection as single feature
    X_proj = projections.reshape(-1, 1)
    y_proj = np.array([0] * min_n + [1] * min_n)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_proj, y_proj, cv=cv, scoring=bal_acc_scorer)
    
    if scores.mean() > best_pca_layer_acc:
        best_pca_layer_acc = scores.mean()
        best_pca_layer = layer_idx

print(f"  Best hallucination direction: layer {best_pca_layer} ({best_pca_layer_acc*100:.1f}%)")

# Multi-layer hallucination direction
X_t_multi = np.hstack([get_layer_features(truth_data, l, min_n) for l in layers_no_emb])
X_h_multi = np.hstack([get_layer_features(hall_data, l, min_n) for l in layers_no_emb])

mean_diff_multi = X_h_multi.mean(axis=0) - X_t_multi.mean(axis=0)
mean_diff_multi = mean_diff_multi / (np.linalg.norm(mean_diff_multi) + 1e-8)

X_all_multi = np.vstack([X_t_multi, X_h_multi])
proj_multi = (X_all_multi @ mean_diff_multi).reshape(-1, 1)
y_multi = np.array([0] * min_n + [1] * min_n)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_multi = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=42),
    proj_multi, y_multi, cv=cv, scoring=bal_acc_scorer
)
print(f"  Multi-layer direction: {scores_multi.mean()*100:.1f}%")

# CMS (Contrast-consistent search) inspired: top PCA components of difference
X_diff_samples = X_h_multi - X_t_multi  # per-sample differences
pca_diff = PCA(n_components=min(10, min_n - 1), random_state=42)
diff_components = pca_diff.fit_transform(StandardScaler().fit_transform(X_diff_samples))

# Project original data onto top difference components
X_all_scaled = StandardScaler().fit_transform(X_all_multi)
X_proj_pca = X_all_scaled @ pca_diff.components_.T
scores_pca_dir = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=42),
    X_proj_pca, y_multi, cv=cv, scoring=bal_acc_scorer
)
print(f"  PCA difference components (top {pca_diff.n_components_}): {scores_pca_dir.mean()*100:.1f}%")

results_all["method5_hallucination_direction"] = {
    "single_layer": {"layer": best_pca_layer, "bal_acc": float(best_pca_layer_acc)},
    "multi_layer": {"bal_acc": float(scores_multi.mean())},
    "pca_components": {"bal_acc": float(scores_pca_dir.mean()), "n_components": int(pca_diff.n_components_)},
}

# ============================================================
# METHOD 6: Permutation test on best method
# ============================================================
print("\n[8/8] Permutation test on best method...")

# Find best overall method
all_methods = {
    "baseline_single_layer": best_single_acc,
    "method1_fusion": best_fusion_acc,
    "method2_layer_diff": best_diff_acc,
    "method3_layer_stats": result_stats["best_bal_acc"],
    "method4_combined": final_combined_acc,
    "method5_direction_single": best_pca_layer_acc,
    "method5_direction_multi": float(scores_multi.mean()),
    "method5_pca_components": float(scores_pca_dir.mean()),
}

best_method_name = max(all_methods, key=all_methods.get)
best_method_acc = all_methods[best_method_name]

print(f"  Best method: {best_method_name} ({best_method_acc*100:.1f}%)")
print(f"  Running 500 permutation tests...")

# Use the best single layer for permutation (simplest, most interpretable)
X_t_perm = get_layer_features(truth_data, best_single_layer, min_n)
X_h_perm = get_layer_features(hall_data, best_single_layer, min_n)
X_perm = np.vstack([X_t_perm, X_h_perm])
y_perm_real = np.array([0] * min_n + [1] * min_n)
X_perm_s = StandardScaler().fit_transform(X_perm)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

null_scores = []
for i in range(500):
    y_shuffled = np.random.permutation(y_perm_real)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_perm_s, y_shuffled, cv=cv, scoring=bal_acc_scorer)
    null_scores.append(scores.mean())
    if (i + 1) % 100 == 0:
        print(f"    ... {i+1}/500 done")

p_value = np.mean([s >= best_single_acc for s in null_scores])
sig_str = "HIGHLY SIGNIFICANT" if p_value < 0.001 else ("SIGNIFICANT" if p_value < 0.05 else "NOT significant")

print(f"  Real: {best_single_acc*100:.1f}%, Null: {np.mean(null_scores)*100:.1f}% +/- {np.std(null_scores)*100:.1f}%")
print(f"  p-value: {p_value:.4f} ({sig_str})")

results_all["permutation_test"] = {
    "real_acc": float(best_single_acc),
    "null_mean": float(np.mean(null_scores)),
    "null_std": float(np.std(null_scores)),
    "p_value": float(p_value),
}

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"STAGE 7 RESULTS SUMMARY")
print(f"{'='*60}")

print(f"\n  Truth vs Hallucination Detection:")
print(f"  {'Method':<35s} {'Bal.Acc':>10s} {'vs Baseline':>12s}")
print(f"  {'-'*60}")

sorted_methods = sorted(all_methods.items(), key=lambda x: x[1], reverse=True)
for name, acc in sorted_methods:
    diff = acc - best_single_acc
    diff_str = f"+{diff*100:.1f}%" if diff > 0 else f"{diff*100:.1f}%"
    marker = " <-- BEST" if acc == best_method_acc else ""
    base = " (BASELINE)" if name == "baseline_single_layer" else ""
    print(f"  {name:<35s} {acc*100:>9.1f}% {diff_str:>11s}{marker}{base}")

print(f"\n  Chance level: 50.0%")
print(f"  Baseline (single layer {best_single_layer}): {best_single_acc*100:.1f}%")
print(f"  Best method: {best_method_name} ({best_method_acc*100:.1f}%)")
print(f"  Improvement over baseline: +{(best_method_acc - best_single_acc)*100:.1f}%")
print(f"  p-value (baseline): {p_value:.4f}")

print(f"\n  INTERPRETATION:")
if best_method_acc > 0.80:
    print(f"    STRONG: Hallucination detection at {best_method_acc*100:.1f}% is commercially viable.")
    print(f"    Advanced methods significantly improve over single-layer baseline.")
elif best_method_acc > 0.70:
    print(f"    GOOD: Hallucination detection at {best_method_acc*100:.1f}% is promising.")
    print(f"    With more data (currently {min_n} per class), accuracy should improve further.")
elif best_method_acc > 0.60:
    print(f"    MODERATE: Hallucination detection at {best_method_acc*100:.1f}%.")
    print(f"    Signal exists but needs more data or different approach for production use.")
else:
    print(f"    WEAK: Hallucination detection at {best_method_acc*100:.1f}% is near chance.")
    print(f"    Hidden states alone may not be sufficient for hallucination detection.")

print(f"\n  COMPARISON WITH ALL STAGES:")
print(f"    Stage 4 (trivia lies):        82.5%")
print(f"    Stage 5 (real-world lies):    70.4%")
print(f"    Stage 6 (lie vs halluc):     100.0%")
print(f"    Stage 6 (truth vs halluc):    67.4% (single layer)")
print(f"    Stage 7 (truth vs halluc):    {best_method_acc*100:.1f}% (best method)")

# Save results
os.makedirs("results", exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

output = {
    "stage": "Stage 7 - Advanced Hallucination Detection",
    "model": model_name,
    "truth_samples": len(truth_data),
    "hallucination_samples": len(hall_data),
    "balanced_n": min_n,
    "baseline_single_layer": {"layer": best_single_layer, "bal_acc": best_single_acc},
    "all_methods": {k: float(v) for k, v in all_methods.items()},
    "best_method": best_method_name,
    "best_acc": float(best_method_acc),
    "permutation_p_value": float(p_value),
    "detailed_results": results_all,
    "elapsed_seconds": time.time() - start_time,
}

with open("results/stage7_results.json", "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)

elapsed = time.time() - start_time
print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Saved: results/stage7_results.json")
print("=" * 60)
print("STAGE 7 COMPLETE")
print("=" * 60)
