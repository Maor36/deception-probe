"""
Thorough validation of probe accuracy on exp02b with AI-judge labels.
Tests multiple methods and random seeds to ensure results are real.
"""

import json
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.dummy import DummyClassifier

# Load data
with open("results/exp02b_relabeled.json") as f:
    data = json.load(f)
responses = data["responses"]
hidden = np.load("results/exp02b_hidden_states.npz")

# AI judge labels
indices = []
labels = []
for i, r in enumerate(responses):
    al = r.get("ai_label", "")
    if al == "LIED":
        indices.append(i)
        labels.append(0)
    elif al == "RESISTED":
        indices.append(i)
        labels.append(1)

labels = np.array(labels)
indices = np.array(indices)
print(f"Samples: {len(labels)} (LIED={np.sum(labels==0)}, RESISTED={np.sum(labels==1)})")

# Test on best layer (14) with multiple approaches
LAYER = 14
X = hidden[f"layer_{LAYER}"][indices]
print(f"\nLayer {LAYER}: X.shape = {X.shape}")
print()

# =============================================
# METHOD 1: Raw features (no dim reduction)
# =============================================
print("=" * 70)
print("METHOD 1: Raw 4096 features, LogReg, 5-fold CV")
print("=" * 70)
for seed in [42, 123, 456, 789, 999]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, C=1.0)
    scores = cross_val_score(clf, X_s, labels, cv=cv, scoring="balanced_accuracy")
    print(f"  seed={seed}: {scores.mean():.3f} ± {scores.std():.3f}  folds={scores}")

# =============================================
# METHOD 2: SVD=64
# =============================================
print()
print("=" * 70)
print("METHOD 2: SVD→64d, LogReg, 5-fold CV")
print("=" * 70)
for seed in [42, 123, 456, 789, 999]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    svd = TruncatedSVD(n_components=64, random_state=seed)
    X_r = svd.fit_transform(X_s)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    scores = cross_val_score(clf, X_r, labels, cv=cv, scoring="balanced_accuracy")
    print(f"  seed={seed}: {scores.mean():.3f} ± {scores.std():.3f}  folds={scores}")

# =============================================
# METHOD 3: PCA=128 (what we used before)
# =============================================
print()
print("=" * 70)
print("METHOD 3: PCA→128d, LogReg, 5-fold CV")
print("=" * 70)
for seed in [42, 123, 456, 789, 999]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=128, random_state=seed)
    X_r = pca.fit_transform(X_s)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    scores = cross_val_score(clf, X_r, labels, cv=cv, scoring="balanced_accuracy")
    print(f"  seed={seed}: {scores.mean():.3f} ± {scores.std():.3f}  folds={scores}")

# =============================================
# BASELINES
# =============================================
print()
print("=" * 70)
print("BASELINES")
print("=" * 70)

# Chance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_val_score(dummy, X, labels, cv=cv, scoring="balanced_accuracy")
print(f"  Chance (most_frequent): {dummy_scores.mean():.3f}")

dummy2 = DummyClassifier(strategy="stratified", random_state=42)
dummy2_scores = cross_val_score(dummy2, X, labels, cv=cv, scoring="balanced_accuracy")
print(f"  Chance (stratified):    {dummy2_scores.mean():.3f}")

# Length baseline
lengths = np.array([len(responses[i].get("phase_b_response", "")) for i in indices]).reshape(-1, 1)
clf_len = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
len_scores = cross_val_score(clf_len, lengths, labels, cv=cv, scoring="balanced_accuracy")
print(f"  Length baseline:        {len_scores.mean():.3f}")

# Shuffled labels (permutation)
print()
print("=" * 70)
print("PERMUTATION TEST (10 shuffles, SVD=64, seed=42)")
print("=" * 70)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
svd = TruncatedSVD(n_components=64, random_state=42)
X_r = svd.fit_transform(X_s)

perm_scores = []
for p in range(10):
    shuffled = labels.copy()
    np.random.seed(p)
    np.random.shuffle(shuffled)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    scores = cross_val_score(clf, X_r, shuffled, cv=cv, scoring="balanced_accuracy")
    perm_scores.append(scores.mean())
    print(f"  Permutation {p}: {scores.mean():.3f}")

print(f"\n  Permutation mean: {np.mean(perm_scores):.3f} ± {np.std(perm_scores):.3f}")
print(f"  Real accuracy:    ~0.72")
print(f"  p-value estimate: {np.mean(np.array(perm_scores) >= 0.72):.3f}")
