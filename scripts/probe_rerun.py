"""
Re-run probe with AI-judge labels. Simple, clear, every layer.
Uses truncated SVD (faster than PCA for sparse/large data).
"""

import json
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import TruncatedSVD

# Load relabeled data
with open("results/exp02b_relabeled.json") as f:
    data = json.load(f)

responses = data["responses"]

# Load hidden states
hidden = np.load("results/exp02b_hidden_states.npz")

# Build label arrays for different label sources
# 1. AI judge labels (LIED vs RESISTED only)
ai_indices = []
ai_labels = []
for i, r in enumerate(responses):
    al = r.get("ai_label", "")
    if al == "LIED":
        ai_indices.append(i)
        ai_labels.append(0)
    elif al == "RESISTED":
        ai_indices.append(i)
        ai_labels.append(1)

ai_labels = np.array(ai_labels)
ai_indices = np.array(ai_indices)

# 2. Original labels
orig_indices = []
orig_labels = []
for i, r in enumerate(responses):
    ol = r.get("label", "")
    if ol == "lied":
        orig_indices.append(i)
        orig_labels.append(0)
    elif ol == "resisted":
        orig_indices.append(i)
        orig_labels.append(1)

orig_labels = np.array(orig_labels)
orig_indices = np.array(orig_indices)

print(f"AI Judge labels: {len(ai_labels)} (LIED={np.sum(ai_labels==0)}, RESISTED={np.sum(ai_labels==1)})")
print(f"Original labels: {len(orig_labels)} (lied={np.sum(orig_labels==0)}, resisted={np.sum(orig_labels==1)})")
print()

# Run probe on every layer for both label sets
layer_keys = sorted(
    [k for k in hidden.files if k.startswith("layer_")],
    key=lambda x: int(x.split("_")[1])
)

N_DIM = 64
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 80)
print(f"LAYER-BY-LAYER PROBE (SVD→{N_DIM}d, 5-fold CV, balanced)")
print(f"{'Layer':>6s} | {'AI Judge':>10s} | {'Original':>10s} | {'Diff':>6s}")
print("-" * 80)

ai_results = {}
orig_results = {}

for key in layer_keys:
    layer_idx = int(key.split("_")[1])
    X_all = hidden[key]
    
    t0 = time.time()
    
    # AI judge labels
    X_ai = X_all[ai_indices]
    scaler = StandardScaler()
    X_ai_s = scaler.fit_transform(X_ai)
    svd = TruncatedSVD(n_components=N_DIM, random_state=42)
    X_ai_r = svd.fit_transform(X_ai_s)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    scores_ai = cross_val_score(clf, X_ai_r, ai_labels, cv=cv, scoring="balanced_accuracy")
    
    # Original labels
    X_orig = X_all[orig_indices]
    scaler2 = StandardScaler()
    X_orig_s = scaler2.fit_transform(X_orig)
    svd2 = TruncatedSVD(n_components=N_DIM, random_state=42)
    X_orig_r = svd2.fit_transform(X_orig_s)
    clf2 = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    scores_orig = cross_val_score(clf2, X_orig_r, orig_labels, cv=cv, scoring="balanced_accuracy")
    
    dt = time.time() - t0
    
    ai_mean = scores_ai.mean()
    orig_mean = scores_orig.mean()
    diff = ai_mean - orig_mean
    
    ai_results[layer_idx] = float(ai_mean)
    orig_results[layer_idx] = float(orig_mean)
    
    tag = ""
    if layer_idx == 0:
        tag = " (EMBEDDING)"
    
    print(f"  {layer_idx:4d} | {ai_mean:9.3f}  | {orig_mean:9.3f}  | {diff:+.3f} | {dt:.1f}s{tag}")

print()
best_ai = max(ai_results, key=ai_results.get)
best_orig = max(orig_results, key=orig_results.get)
print(f"BEST AI Judge:  Layer {best_ai} = {ai_results[best_ai]:.3f}")
print(f"BEST Original:  Layer {best_orig} = {orig_results[best_orig]:.3f}")

# Length baseline for both
print("\n--- Length Baselines ---")
for name, indices, labels in [("AI Judge", ai_indices, ai_labels), ("Original", orig_indices, orig_labels)]:
    lengths = np.array([len(responses[i].get("phase_b_response", "")) for i in indices]).reshape(-1, 1)
    clf_len = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    len_scores = cross_val_score(clf_len, lengths, labels, cv=5, scoring="balanced_accuracy")
    print(f"  {name:12s}: {len_scores.mean():.3f}")
