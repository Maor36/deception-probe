"""
Quick probe analysis using AI-judge labels (LIED vs RESISTED only, excluding REFUSED).
Uses PCA for dimensionality reduction to speed up computation.
"""

import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load relabeled data
with open("results/exp02b_relabeled.json") as f:
    data = json.load(f)

responses = data["responses"]

# Load hidden states
hidden = np.load("results/exp02b_hidden_states.npz")

# Filter: only LIED and RESISTED (exclude REFUSED and ERROR)
valid_indices = []
labels = []
for i, r in enumerate(responses):
    ai_label = r.get("ai_label", "")
    if ai_label == "LIED":
        valid_indices.append(i)
        labels.append(0)
    elif ai_label == "RESISTED":
        valid_indices.append(i)
        labels.append(1)

labels = np.array(labels)
print(f"Valid samples: {len(labels)}")
print(f"  LIED (0): {np.sum(labels == 0)}")
print(f"  RESISTED (1): {np.sum(labels == 1)}")
print()

# Per-layer probe with PCA
layer_keys = sorted(
    [k for k in hidden.files if k.startswith("layer_")],
    key=lambda x: int(x.split("_")[1])
)

N_COMPONENTS = 128  # Reduce from 4096 to 128

print("=" * 60)
print(f"PER-LAYER PROBE ACCURACY (AI-judge labels, PCA={N_COMPONENTS})")
print("=" * 60)

best_layer = -1
best_acc = 0.0
results = {}

for key in layer_keys:
    layer_idx = int(key.split("_")[1])
    X = hidden[key][valid_indices]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS, random_state=42)),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        )),
    ])

    scores = cross_val_score(pipe, X, labels, cv=5, scoring="balanced_accuracy")
    mean_acc = scores.mean()
    std_acc = scores.std()

    results[layer_idx] = {"mean": float(mean_acc), "std": float(std_acc)}

    if mean_acc > best_acc:
        best_acc = mean_acc
        best_layer = layer_idx

    tag = ""
    if layer_idx == 0:
        tag = " (EMBEDDING)"
    elif layer_idx == best_layer and layer_idx > 0:
        tag = " *** BEST ***"

    print(f"  Layer {layer_idx:2d}: {mean_acc:.3f} ± {std_acc:.3f}{tag}")

print(f"\n  BEST LAYER: {best_layer} ({best_acc:.3f})")

# Length baseline
print("\n--- Length Baseline ---")
lengths = np.array([len(responses[i].get("phase_b_response", "")) for i in valid_indices])
X_len = lengths.reshape(-1, 1)
pipe_len = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
len_scores = cross_val_score(pipe_len, X_len, labels, cv=5, scoring="balanced_accuracy")
print(f"  Length baseline: {len_scores.mean():.3f}")

# Per pressure type
print("\n--- Per Pressure Type ---")
pressure_types = np.array([responses[i].get("pressure_type", "?") for i in valid_indices])
for ptype in np.unique(pressure_types):
    mask = pressure_types == ptype
    n = mask.sum()
    n_lied = np.sum(labels[mask] == 0)
    n_resisted = np.sum(labels[mask] == 1)
    lie_rate = n_lied / n if n > 0 else 0
    print(f"  {ptype:20s} | n={n:3d} | LIED={n_lied:3d} | RESISTED={n_resisted:3d} | lie_rate={lie_rate:.1%}")

# Save results
output = {
    "best_layer": best_layer,
    "best_accuracy": float(best_acc),
    "length_baseline": float(len_scores.mean()),
    "n_lied": int(np.sum(labels == 0)),
    "n_resisted": int(np.sum(labels == 1)),
    "n_refused": 31,
    "pca_components": N_COMPONENTS,
    "layer_results": {str(k): v for k, v in results.items()},
    "label_source": "gpt-4.1-nano AI judge",
}
with open("results/exp02c_probe_results_aijudge.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → results/exp02c_probe_results_aijudge.json")
