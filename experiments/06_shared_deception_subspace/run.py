"""
Experiment 06 — Shared Deception Subspace
==========================================
PURPOSE:
    Experiment 05 showed that different deception types (sycophancy,
    instruction conflict, authority pressure) have nearly orthogonal
    lie directions. But does that mean there is NO shared structure?

    This experiment investigates whether a shared deception subspace
    exists using same-layer comparisons, dimensionality reduction,
    multi-task probing, and subspace analysis.

PREREQUISITE:
    Must run exp05 v3+ first (saves vectors from ALL probe layers).
    Requires: results/exp05_vectors.npz

CRITICAL FIX (v2):
    All analyses now compare vectors from the SAME layer. Previous
    version mixed vectors from different layers (16/18/20), which
    confounded layer differences with deception type differences.

ANALYSES:
    1. PER-LAYER PCA — Project all types from same layer, check separation
    2. PER-LAYER SHARED PROBE — Train on all types from same layer
    3. PER-LAYER SUBSPACE OVERLAP — Principal angles between same-layer subspaces
    4. PER-LAYER PROCRUSTES — Align and transfer within same layer
    5. PER-LAYER LIE DIRECTION RANK — SVD on same-layer lie directions

MODEL:   N/A (works on saved vectors, no GPU needed)
RUNTIME: ~5 minutes on CPU

USAGE:
    %run experiments/06_shared_deception_subspace/run.py

CHANGELOG:
    v2 (2026-03-21): CRITICAL FIX — all analyses now use same-layer
        vectors. Previous version mixed layers, confounding results.
    v1 (2026-03-21): Initial version (had layer-mixing bug).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import json
import time
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes, svd, subspace_angles

from src.utils import setup_logger, save_results

# ── Configuration ──────────────────────────────────────────────────────────
VECTORS_PATH = "results/exp05_vectors.npz"
RANDOM_SEED = 42
PCA_SUBSPACE_DIMS = 10  # For subspace angle analysis

np.random.seed(RANDOM_SEED)
log = setup_logger("exp06")

DECEPTION_TYPES = ["sycophancy", "instruction_conflict", "authority_pressure"]
TYPE_LABELS = {
    "sycophancy": "Sycophancy",
    "instruction_conflict": "Instruction Conflict",
    "authority_pressure": "Authority Pressure",
}


# ── Helper functions ──────────────────────────────────────────────────────

def load_vectors():
    """Load saved vectors from exp05 v3+."""
    if not os.path.exists(VECTORS_PATH):
        raise FileNotFoundError(
            f"{VECTORS_PATH} not found. Run exp05 v3+ first."
        )
    data = np.load(VECTORS_PATH)

    # Check for v3 format (has per-layer keys like "sycophancy_lied_L16")
    has_layer_keys = any("_L" in k for k in data.files)
    if not has_layer_keys:
        raise ValueError(
            "exp05_vectors.npz is in old format (single best-layer only). "
            "Re-run exp05 v3 to save vectors from all layers."
        )

    probe_layers = data["probe_layers"].tolist()
    log.info(f"Loaded vectors from {VECTORS_PATH}")
    log.info(f"  Probe layers: {probe_layers}")
    log.info(f"  Total arrays: {len(data.files)}")
    return data, probe_layers


def get_layer_data(data, layer):
    """Extract lied/resisted vectors for all types from a specific layer."""
    layer_data = {}
    for dtype in DECEPTION_TYPES:
        lied_key = f"{dtype}_lied_L{layer}"
        resisted_key = f"{dtype}_resisted_L{layer}"
        if lied_key in data.files and resisted_key in data.files:
            layer_data[dtype] = {
                "lied": data[lied_key],
                "resisted": data[resisted_key],
            }
    return layer_data


def balanced_acc_scorer():
    return make_scorer(balanced_accuracy_score)


# ── Analysis 1: Per-Layer PCA ─────────────────────────────────────────────

def analysis_pca_per_layer(data, probe_layers):
    """For each layer, project all types and check lied/resisted separation."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 1: Per-Layer PCA")
    log.info("=" * 60)

    results = {}

    for layer in probe_layers:
        layer_data = get_layer_data(data, layer)
        if len(layer_data) < 2:
            continue

        # Collect all vectors from this layer
        all_vecs, all_labels, all_type_ids = [], [], []
        for i, dtype in enumerate(DECEPTION_TYPES):
            if dtype not in layer_data:
                continue
            lied = layer_data[dtype]["lied"]
            resisted = layer_data[dtype]["resisted"]
            all_vecs.extend([lied, resisted])
            all_labels.extend([1]*len(lied) + [0]*len(resisted))
            all_type_ids.extend([i]*len(lied) + [i]*len(resisted))

        X = np.vstack(all_vecs)
        y = np.array(all_labels)
        type_ids = np.array(all_type_ids)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(50, X.shape[0]-1))
        X_pca = pca.fit_transform(X_scaled)

        layer_result = {"n_samples": len(X)}

        # Lied vs Resisted classification in PCA space
        for n_pcs in [2, 5, 10, 20]:
            if n_pcs > X_pca.shape[1]:
                continue
            pipe = Pipeline([
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
            ])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(pipe, X_pca[:, :n_pcs], y, cv=cv, scoring=balanced_acc_scorer())
            layer_result[f"lied_vs_resisted_{n_pcs}pcs"] = {
                "accuracy": float(scores.mean()),
                "std": float(scores.std()),
            }

        # Type classification (should be LOW if same-layer vectors are similar)
        pipe = Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X_pca[:, :10], type_ids, cv=cv, scoring="accuracy")
        layer_result["type_classification_10pcs"] = {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
        }

        results[str(layer)] = layer_result

        lie_acc_20 = layer_result.get("lied_vs_resisted_20pcs", {}).get("accuracy", 0)
        type_acc = layer_result["type_classification_10pcs"]["accuracy"]
        log.info(f"  Layer {layer}: lie detect (20 PCs)={lie_acc_20*100:.1f}%, "
                 f"type classify={type_acc*100:.1f}%")

    return results


# ── Analysis 2: Per-Layer Shared Probe ────────────────────────────────────

def analysis_shared_probe_per_layer(data, probe_layers):
    """For each layer, train one probe on all types combined."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 2: Per-Layer Shared Probe")
    log.info("=" * 60)

    results = {}

    for layer in probe_layers:
        layer_data = get_layer_data(data, layer)
        if len(layer_data) < 2:
            continue

        # All types combined
        all_vecs, all_labels, all_type_ids = [], [], []
        for i, dtype in enumerate(DECEPTION_TYPES):
            if dtype not in layer_data:
                continue
            lied = layer_data[dtype]["lied"]
            resisted = layer_data[dtype]["resisted"]
            all_vecs.extend([lied, resisted])
            all_labels.extend([1]*len(lied) + [0]*len(resisted))
            all_type_ids.extend([i]*len(lied) + [i]*len(resisted))

        X = np.vstack(all_vecs)
        y = np.array(all_labels)
        type_ids = np.array(all_type_ids)

        # Combined probe
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=balanced_acc_scorer())

        layer_result = {
            "combined_accuracy": float(scores.mean()),
            "combined_std": float(scores.std()),
        }

        # Leave-one-type-out
        for test_type_id, test_type_name in enumerate(DECEPTION_TYPES):
            if test_type_name not in layer_data:
                continue
            train_mask = type_ids != test_type_id
            test_mask = type_ids == test_type_id
            if test_mask.sum() < 5 or train_mask.sum() < 10:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
            ])
            pipe.fit(X[train_mask], y[train_mask])
            y_pred = pipe.predict(X[test_mask])
            acc = balanced_accuracy_score(y[test_mask], y_pred)
            layer_result[f"leave_out_{test_type_name}"] = float(acc)

        results[str(layer)] = layer_result

        combined = layer_result["combined_accuracy"]
        leave_outs = {k: v for k, v in layer_result.items() if k.startswith("leave_out_")}
        leave_str = ", ".join(f"{k.replace('leave_out_','')}={v*100:.1f}%" for k, v in leave_outs.items())
        log.info(f"  Layer {layer}: combined={combined*100:.1f}%, {leave_str}")

    return results


# ── Analysis 3: Per-Layer Subspace Overlap ────────────────────────────────

def analysis_subspace_overlap_per_layer(data, probe_layers):
    """Compute principal angles between type subspaces within each layer."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 3: Per-Layer Subspace Overlap (Principal Angles)")
    log.info("=" * 60)

    results = {}

    for layer in probe_layers:
        layer_data = get_layer_data(data, layer)
        if len(layer_data) < 2:
            continue

        # Compute PCA subspace for each type at this layer
        subspaces = {}
        for dtype in DECEPTION_TYPES:
            if dtype not in layer_data:
                continue
            lied = layer_data[dtype]["lied"]
            resisted = layer_data[dtype]["resisted"]
            X = np.vstack([lied, resisted])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_comp = min(PCA_SUBSPACE_DIMS, X.shape[0]-1)
            pca = PCA(n_components=n_comp)
            pca.fit(X_scaled)
            subspaces[dtype] = pca.components_.T  # (4096, n_comp)

        # Principal angles between all pairs
        layer_result = {}
        types_list = [t for t in DECEPTION_TYPES if t in subspaces]
        for i in range(len(types_list)):
            for j in range(i+1, len(types_list)):
                t1, t2 = types_list[i], types_list[j]
                angles = subspace_angles(subspaces[t1], subspaces[t2])
                angles_deg = np.degrees(angles)
                layer_result[f"{t1}_vs_{t2}"] = {
                    "min_angle": float(angles_deg.min()),
                    "mean_angle": float(angles_deg.mean()),
                    "max_angle": float(angles_deg.max()),
                    "angles": angles_deg.tolist(),
                }

        results[str(layer)] = layer_result

        # Log summary
        for pair, vals in layer_result.items():
            status = "SHARED" if vals["min_angle"] < 30 else "PARTIAL" if vals["min_angle"] < 60 else "INDEPENDENT"
            log.info(f"  Layer {layer} {pair}: min={vals['min_angle']:.1f}°, "
                     f"mean={vals['mean_angle']:.1f}° → {status}")

    return results


# ── Analysis 4: Per-Layer Procrustes Transfer ─────────────────────────────

def analysis_procrustes_per_layer(data, probe_layers):
    """Align type spaces with Procrustes within each layer."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 4: Per-Layer Procrustes-Aligned Transfer")
    log.info("=" * 60)

    results = {}

    for layer in probe_layers:
        layer_data = get_layer_data(data, layer)
        if len(layer_data) < 2:
            continue

        layer_result = {}
        types_list = [t for t in DECEPTION_TYPES if t in layer_data]

        for src_type in types_list:
            for tgt_type in types_list:
                if src_type == tgt_type:
                    continue

                src_lied = layer_data[src_type]["lied"]
                src_resisted = layer_data[src_type]["resisted"]
                tgt_lied = layer_data[tgt_type]["lied"]
                tgt_resisted = layer_data[tgt_type]["resisted"]

                X_src = np.vstack([src_lied, src_resisted])
                y_src = np.array([1]*len(src_lied) + [0]*len(src_resisted))
                X_tgt = np.vstack([tgt_lied, tgt_resisted])
                y_tgt = np.array([1]*len(tgt_lied) + [0]*len(tgt_resisted))

                # Reduce dims for Procrustes
                n_dims = min(50, X_src.shape[0]-1, X_tgt.shape[0]-1)
                scaler_src = StandardScaler()
                scaler_tgt = StandardScaler()
                X_src_s = scaler_src.fit_transform(X_src)
                X_tgt_s = scaler_tgt.fit_transform(X_tgt)

                pca_src = PCA(n_components=n_dims)
                pca_tgt = PCA(n_components=n_dims)
                X_src_pca = pca_src.fit_transform(X_src_s)
                X_tgt_pca = pca_tgt.fit_transform(X_tgt_s)

                # Baseline: no alignment
                pipe = Pipeline([
                    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
                ])
                pipe.fit(X_src_pca, y_src)
                baseline_acc = balanced_accuracy_score(y_tgt, pipe.predict(X_tgt_pca))

                # Procrustes alignment using class centroids
                src_centroids = np.array([X_src_pca[y_src==0].mean(0), X_src_pca[y_src==1].mean(0)])
                tgt_centroids = np.array([X_tgt_pca[y_tgt==0].mean(0), X_tgt_pca[y_tgt==1].mean(0)])
                R, _ = orthogonal_procrustes(src_centroids, tgt_centroids)

                X_tgt_aligned = X_tgt_pca @ R
                pipe_aligned = Pipeline([
                    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
                ])
                pipe_aligned.fit(X_src_pca, y_src)
                aligned_acc = balanced_accuracy_score(y_tgt, pipe_aligned.predict(X_tgt_aligned))

                layer_result[f"{src_type}->{tgt_type}"] = {
                    "baseline": float(baseline_acc),
                    "aligned": float(aligned_acc),
                    "improvement": float(aligned_acc - baseline_acc),
                }

        results[str(layer)] = layer_result

        # Log best improvements
        if layer_result:
            improvements = [(k, v["improvement"]) for k, v in layer_result.items()]
            avg_imp = np.mean([imp for _, imp in improvements])
            best_pair, best_imp = max(improvements, key=lambda x: x[1])
            log.info(f"  Layer {layer}: avg improvement={avg_imp*100:+.1f}%, "
                     f"best={best_pair} ({best_imp*100:+.1f}%)")

    return results


# ── Analysis 5: Per-Layer Lie Direction Rank ──────────────────────────────

def analysis_rank_per_layer(data, probe_layers):
    """Check if lie directions are independent at each layer."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 5: Per-Layer Lie Direction Rank")
    log.info("=" * 60)

    results = {}

    for layer in probe_layers:
        directions = []
        names = []
        for dtype in DECEPTION_TYPES:
            key = f"lie_dir_{dtype}_L{layer}"
            if key in data.files:
                directions.append(data[key])
                names.append(dtype)

        if len(directions) < 2:
            continue

        D = np.array(directions)
        U, S, Vt = svd(D, full_matrices=False)
        relative = S / S[0]

        # Pairwise cosines at this layer
        cosines = {}
        for i in range(len(directions)):
            for j in range(i+1, len(directions)):
                cos = float(np.dot(directions[i], directions[j]))
                cosines[f"{names[i]}_vs_{names[j]}"] = cos

        effective_rank = int(np.sum(relative > 0.1))

        results[str(layer)] = {
            "singular_values": S.tolist(),
            "relative_magnitudes": relative.tolist(),
            "effective_rank": effective_rank,
            "pairwise_cosines": cosines,
        }

        avg_cos = np.mean(list(cosines.values())) if cosines else 0
        log.info(f"  Layer {layer}: rank={effective_rank}/3, "
                 f"SVs={S.round(3)}, avg_cosine={avg_cos:.4f}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 06: Shared Deception Subspace (v2 — same-layer fix)")
    log.info("=" * 60)
    start = time.time()

    data, probe_layers = load_vectors()

    # Run all analyses
    pca_results = analysis_pca_per_layer(data, probe_layers)
    shared_probe_results = analysis_shared_probe_per_layer(data, probe_layers)
    subspace_results = analysis_subspace_overlap_per_layer(data, probe_layers)
    procrustes_results = analysis_procrustes_per_layer(data, probe_layers)
    rank_results = analysis_rank_per_layer(data, probe_layers)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SUMMARY — Best results across layers")
    log.info("=" * 60)

    # Best combined probe
    best_combined = 0
    best_combined_layer = None
    for layer_str, res in shared_probe_results.items():
        acc = res.get("combined_accuracy", 0)
        if acc > best_combined:
            best_combined = acc
            best_combined_layer = layer_str
    log.info(f"  Best combined probe: {best_combined*100:.1f}% (Layer {best_combined_layer})")

    # Best leave-one-out
    for dtype in DECEPTION_TYPES:
        best_loo = 0
        best_loo_layer = None
        for layer_str, res in shared_probe_results.items():
            acc = res.get(f"leave_out_{dtype}", 0)
            if acc > best_loo:
                best_loo = acc
                best_loo_layer = layer_str
        log.info(f"  Best leave-out {TYPE_LABELS[dtype]}: {best_loo*100:.1f}% (Layer {best_loo_layer})")

    # Best Procrustes improvement
    best_proc_imp = -1
    best_proc_pair = ""
    best_proc_layer = ""
    for layer_str, layer_res in procrustes_results.items():
        for pair, vals in layer_res.items():
            if vals["improvement"] > best_proc_imp:
                best_proc_imp = vals["improvement"]
                best_proc_pair = pair
                best_proc_layer = layer_str
    log.info(f"  Best Procrustes improvement: {best_proc_imp*100:+.1f}% "
             f"({best_proc_pair}, Layer {best_proc_layer})")

    # Smallest principal angle
    smallest_angle = 90
    smallest_pair = ""
    smallest_layer = ""
    for layer_str, layer_res in subspace_results.items():
        for pair, vals in layer_res.items():
            if vals["min_angle"] < smallest_angle:
                smallest_angle = vals["min_angle"]
                smallest_pair = pair
                smallest_layer = layer_str
    log.info(f"  Smallest principal angle: {smallest_angle:.1f}° "
             f"({smallest_pair}, Layer {smallest_layer})")

    # Conclusion
    if best_combined > 0.65:
        log.info("\n  CONCLUSION: A shared deception subspace EXISTS.")
        log.info("  When comparing from the SAME layer, different deception types")
        log.info("  share detectable internal structure.")
    elif best_combined > 0.55:
        log.info("\n  CONCLUSION: WEAK shared structure detected.")
    else:
        log.info("\n  CONCLUSION: No shared deception subspace found.")

    # Save
    output = {
        "experiment": "06_shared_deception_subspace",
        "version": "v2_same_layer_fix",
        "results": {
            "pca_per_layer": pca_results,
            "shared_probe_per_layer": shared_probe_results,
            "subspace_overlap_per_layer": subspace_results,
            "procrustes_per_layer": procrustes_results,
            "lie_direction_rank_per_layer": rank_results,
        },
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp06_shared_subspace.json")
    log.info(f"\nSaved to results/exp06_shared_subspace.json")
    log.info(f"Total time: {time.time() - start:.0f}s")

    return output


if __name__ == "__main__":
    main()
