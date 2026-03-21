"""
Experiment 06 — Shared Deception Subspace
==========================================
PURPOSE:
    Experiment 05 showed that different deception types (sycophancy,
    instruction conflict, authority pressure) have nearly orthogonal
    lie directions (cosine ~0.01-0.04). But does that mean there is
    NO shared structure at all?

    This experiment investigates whether a shared deception subspace
    exists when we look beyond single directions — using dimensionality
    reduction, multi-task probing, and subspace analysis.

PREREQUISITE:
    Must run exp05 first with vector saving enabled.
    Requires: results/exp05_vectors.npz

ANALYSES:
    1. PCA VISUALIZATION
       - Project all vectors (all 3 types) into shared PCA space
       - Check if lied/resisted clusters separate in top PCs
       - Visualize 2D and 3D projections

    2. SHARED SUBSPACE PROBE (Multi-task)
       - Train a single probe on ALL deception types combined
       - If accuracy > chance, there IS a shared signal
       - Compare to within-type accuracy (how much is lost?)

    3. SUBSPACE OVERLAP (Principal Angles)
       - For each deception type, find the top-k PCA subspace of lied vs resisted
       - Compute principal angles between subspaces
       - Small angles = shared structure, 90° = fully independent

    4. CROSS-TYPE TRANSFER WITH ALIGNMENT
       - Use Procrustes alignment to map one type's space to another
       - Re-test cross-type transfer after alignment
       - If transfer improves, the structure IS shared but rotated

    5. CONCATENATED LIE DIRECTION ANALYSIS
       - Stack all 3 lie direction vectors
       - SVD to find if they span a low-rank subspace
       - If rank < 3, there's redundancy (shared structure)

MODEL:   N/A (works on saved vectors, no GPU needed)
RUNTIME: ~2 minutes on CPU

USAGE:
    %run experiments/06_shared_deception_subspace/run.py

CHANGELOG:
    v1 (2026-03-21): Initial version.
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
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes, svd, subspace_angles

from src.utils import setup_logger, save_results, permutation_test

# ── Configuration ──────────────────────────────────────────────────────────
VECTORS_PATH = "results/exp05_vectors.npz"
N_PERMUTATIONS = 200
RANDOM_SEED = 42
PCA_DIMS = 50  # For subspace analysis
SAVE_PLOTS = True

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
    """Load saved vectors from exp05."""
    if not os.path.exists(VECTORS_PATH):
        raise FileNotFoundError(
            f"{VECTORS_PATH} not found. Run exp05 first with vector saving enabled."
        )
    data = np.load(VECTORS_PATH)
    log.info(f"Loaded vectors from {VECTORS_PATH}")
    for key in data.files:
        log.info(f"  {key}: {data[key].shape}")
    return data


def balanced_acc_scorer():
    return make_scorer(balanced_accuracy_score)


# ── Analysis 1: PCA Visualization ─────────────────────────────────────────

def analysis_pca_visualization(data):
    """Project all vectors into shared PCA space and check separation."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 1: PCA Visualization")
    log.info("=" * 60)

    # Collect all vectors with labels
    all_vecs = []
    all_labels = []      # 0=resisted, 1=lied
    all_types = []       # deception type name
    all_type_ids = []    # numeric type id

    for i, dtype in enumerate(DECEPTION_TYPES):
        lied_key = f"{dtype}_lied"
        resisted_key = f"{dtype}_resisted"
        if lied_key not in data.files or resisted_key not in data.files:
            log.warning(f"  Skipping {dtype}: vectors not found")
            continue

        lied = data[lied_key]
        resisted = data[resisted_key]

        all_vecs.append(lied)
        all_labels.extend([1] * len(lied))
        all_types.extend([dtype] * len(lied))
        all_type_ids.extend([i] * len(lied))

        all_vecs.append(resisted)
        all_labels.extend([0] * len(resisted))
        all_types.extend([dtype] * len(resisted))
        all_type_ids.extend([i] * len(resisted))

    X = np.vstack(all_vecs)
    y = np.array(all_labels)
    type_ids = np.array(all_type_ids)

    log.info(f"  Total vectors: {X.shape[0]} x {X.shape[1]}")
    log.info(f"  Lied: {sum(y==1)}, Resisted: {sum(y==0)}")

    # Fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    log.info(f"  Explained variance (top 10 PCs): {pca.explained_variance_ratio_[:10].round(4)}")
    log.info(f"  Cumulative variance (10 PCs): {pca.explained_variance_ratio_[:10].sum():.4f}")
    log.info(f"  Cumulative variance (20 PCs): {pca.explained_variance_ratio_[:20].sum():.4f}")

    # Check: can we separate lied/resisted in PCA space?
    for n_pcs in [2, 5, 10, 20]:
        pipe = Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X_pca[:, :n_pcs], y, cv=cv, scoring=balanced_acc_scorer())
        log.info(f"  Lied vs Resisted in {n_pcs} PCs (all types mixed): {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Check: can we separate deception TYPES in PCA space?
    for n_pcs in [2, 5, 10]:
        pipe = Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, multi_class="multinomial")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X_pca[:, :n_pcs], type_ids, cv=cv, scoring="accuracy")
        log.info(f"  Type classification in {n_pcs} PCs: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # Save PCA coordinates for plotting
    if SAVE_PLOTS:
        np.savez_compressed(
            "results/exp06_pca_coords.npz",
            X_pca=X_pca,
            labels=y,
            type_ids=type_ids,
            explained_variance=pca.explained_variance_ratio_,
        )
        log.info("  Saved PCA coordinates to results/exp06_pca_coords.npz")

    return X, y, type_ids, X_pca, pca


# ── Analysis 2: Multi-task Shared Probe ───────────────────────────────────

def analysis_shared_probe(X, y, type_ids):
    """Train a single probe on all deception types combined."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 2: Shared Probe (Multi-task)")
    log.info("=" * 60)

    results = {}

    # A) Single probe on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pipe = Pipeline([
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X_scaled, y, cv=cv, scoring=balanced_acc_scorer())
    results["all_types_combined"] = {
        "accuracy": float(scores.mean()),
        "std": float(scores.std()),
    }
    log.info(f"  All types combined: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    # B) Leave-one-type-out: train on 2 types, test on 3rd
    for test_type_id, test_type_name in enumerate(DECEPTION_TYPES):
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

        results[f"leave_out_{test_type_name}"] = {
            "accuracy": float(acc),
            "train_types": [t for i, t in enumerate(DECEPTION_TYPES) if i != test_type_id],
            "test_type": test_type_name,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
        log.info(f"  Train on others, test on {TYPE_LABELS[test_type_name]}: {acc*100:.1f}%")

    # C) Permutation test on combined probe
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pipe = Pipeline([
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])
    pipe.fit(X_scaled, y)
    y_pred = pipe.predict(X_scaled)
    observed_acc = balanced_accuracy_score(y, y_pred)
    perm = permutation_test(X, y, observed_acc, N_PERMUTATIONS, random_seed=RANDOM_SEED)
    results["permutation_test"] = {
        "observed": float(observed_acc),
        "p_value": float(perm["p_value"]),
    }
    log.info(f"  Combined probe permutation test: p={perm['p_value']:.4f}")

    return results


# ── Analysis 3: Subspace Overlap (Principal Angles) ───────────────────────

def analysis_subspace_overlap(data):
    """Compute principal angles between deception type subspaces."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 3: Subspace Overlap (Principal Angles)")
    log.info("=" * 60)

    results = {}

    # For each type, compute the subspace that separates lied from resisted
    subspaces = {}
    for dtype in DECEPTION_TYPES:
        lied_key = f"{dtype}_lied"
        resisted_key = f"{dtype}_resisted"
        if lied_key not in data.files:
            continue

        lied = data[lied_key]
        resisted = data[resisted_key]

        # Compute difference vectors (lied - resisted mean)
        X = np.vstack([lied, resisted])
        y = np.array([1]*len(lied) + [0]*len(resisted))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get top-k PCA directions of the lied-resisted difference
        pca = PCA(n_components=min(PCA_DIMS, len(X)-1))
        X_pca = pca.fit_transform(X_scaled)

        # Train probe in PCA space to find discriminative subspace
        pipe = Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
        ])
        pipe.fit(X_pca, y)

        # The probe's weight vector in PCA space, projected back to original
        coef_pca = pipe.named_steps["clf"].coef_[0]
        # Top discriminative directions in original space
        discriminative_dir = pca.components_.T @ coef_pca
        discriminative_dir = discriminative_dir / np.linalg.norm(discriminative_dir)

        # Store the top PCA components as the subspace
        subspaces[dtype] = pca.components_[:10].T  # (4096, 10) — top 10 PCs

        log.info(f"  {TYPE_LABELS[dtype]}: subspace computed (top 10 PCs, "
                 f"variance explained: {pca.explained_variance_ratio_[:10].sum():.3f})")

    # Compute principal angles between all pairs
    for i in range(len(DECEPTION_TYPES)):
        for j in range(i + 1, len(DECEPTION_TYPES)):
            t1, t2 = DECEPTION_TYPES[i], DECEPTION_TYPES[j]
            if t1 not in subspaces or t2 not in subspaces:
                continue

            angles = subspace_angles(subspaces[t1], subspaces[t2])
            angles_deg = np.degrees(angles)

            results[f"{t1}_vs_{t2}"] = {
                "principal_angles_deg": angles_deg.tolist(),
                "min_angle": float(angles_deg.min()),
                "max_angle": float(angles_deg.max()),
                "mean_angle": float(angles_deg.mean()),
                "median_angle": float(np.median(angles_deg)),
            }

            log.info(f"  {TYPE_LABELS[t1]} vs {TYPE_LABELS[t2]}:")
            log.info(f"    Angles: {angles_deg.round(1)}")
            log.info(f"    Min: {angles_deg.min():.1f}°, Mean: {angles_deg.mean():.1f}°, Max: {angles_deg.max():.1f}°")
            if angles_deg.min() < 30:
                log.info(f"    → SHARED STRUCTURE detected (min angle < 30°)")
            elif angles_deg.min() < 60:
                log.info(f"    → PARTIAL overlap (min angle 30-60°)")
            else:
                log.info(f"    → INDEPENDENT subspaces (min angle > 60°)")

    return results


# ── Analysis 4: Procrustes-Aligned Cross-Type Transfer ────────────────────

def analysis_procrustes_transfer(data):
    """Align deception type spaces with Procrustes and re-test transfer."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 4: Procrustes-Aligned Cross-Type Transfer")
    log.info("=" * 60)

    results = {}

    for i in range(len(DECEPTION_TYPES)):
        for j in range(len(DECEPTION_TYPES)):
            if i == j:
                continue

            src_type = DECEPTION_TYPES[i]
            tgt_type = DECEPTION_TYPES[j]

            src_lied = data[f"{src_type}_lied"]
            src_resisted = data[f"{src_type}_resisted"]
            tgt_lied = data[f"{tgt_type}_lied"]
            tgt_resisted = data[f"{tgt_type}_resisted"]

            # Build source and target datasets
            X_src = np.vstack([src_lied, src_resisted])
            y_src = np.array([1]*len(src_lied) + [0]*len(src_resisted))
            X_tgt = np.vstack([tgt_lied, tgt_resisted])
            y_tgt = np.array([1]*len(tgt_lied) + [0]*len(tgt_resisted))

            # Standardize
            scaler_src = StandardScaler()
            X_src_s = scaler_src.fit_transform(X_src)
            scaler_tgt = StandardScaler()
            X_tgt_s = scaler_tgt.fit_transform(X_tgt)

            # Reduce dimensionality for Procrustes (needs square or fewer dims than samples)
            n_dims = min(50, X_src_s.shape[0], X_tgt_s.shape[0])
            pca_src = PCA(n_components=n_dims)
            pca_tgt = PCA(n_components=n_dims)
            X_src_pca = pca_src.fit_transform(X_src_s)
            X_tgt_pca = pca_tgt.fit_transform(X_tgt_s)

            # Baseline: train on source, test on target (no alignment)
            pipe = Pipeline([
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
            ])
            pipe.fit(X_src_pca, y_src)
            baseline_acc = balanced_accuracy_score(y_tgt, pipe.predict(X_tgt_pca))

            # Procrustes alignment: find rotation R that maps src means to tgt means
            # Use class centroids as anchor points
            src_centroids = np.array([X_src_pca[y_src==0].mean(0), X_src_pca[y_src==1].mean(0)])
            tgt_centroids = np.array([X_tgt_pca[y_tgt==0].mean(0), X_tgt_pca[y_tgt==1].mean(0)])

            R, scale = orthogonal_procrustes(src_centroids, tgt_centroids)

            # Apply rotation to source-trained probe predictions
            X_tgt_aligned = X_tgt_pca @ R
            pipe_aligned = Pipeline([
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
            ])
            pipe_aligned.fit(X_src_pca, y_src)
            aligned_acc = balanced_accuracy_score(y_tgt, pipe_aligned.predict(X_tgt_aligned))

            results[f"{src_type}->{tgt_type}"] = {
                "baseline_acc": float(baseline_acc),
                "aligned_acc": float(aligned_acc),
                "improvement": float(aligned_acc - baseline_acc),
            }

            improvement = "↑" if aligned_acc > baseline_acc else "↓" if aligned_acc < baseline_acc else "="
            log.info(f"  {TYPE_LABELS[src_type]} → {TYPE_LABELS[tgt_type]}: "
                     f"baseline={baseline_acc*100:.1f}%, aligned={aligned_acc*100:.1f}% {improvement}")

    return results


# ── Analysis 5: Lie Direction SVD (Rank Analysis) ─────────────────────────

def analysis_lie_direction_rank(data):
    """Check if the 3 lie directions span a low-rank subspace."""
    log.info("\n" + "=" * 60)
    log.info("Analysis 5: Lie Direction Rank Analysis")
    log.info("=" * 60)

    results = {}

    # Stack lie direction vectors
    directions = []
    names = []
    for dtype in DECEPTION_TYPES:
        key = f"lie_dir_{dtype}"
        if key in data.files:
            directions.append(data[key])
            names.append(dtype)

    if len(directions) < 2:
        log.warning("  Not enough lie directions found")
        return results

    D = np.array(directions)  # (3, 4096)
    log.info(f"  Lie direction matrix: {D.shape}")

    # SVD
    U, S, Vt = svd(D, full_matrices=False)
    log.info(f"  Singular values: {S.round(4)}")
    log.info(f"  Relative magnitudes: {(S / S[0]).round(4)}")

    # Effective rank
    total_var = (S**2).sum()
    cumvar = np.cumsum(S**2) / total_var
    log.info(f"  Cumulative variance: {cumvar.round(4)}")

    if S[1] / S[0] < 0.1:
        log.info("  → RANK 1: All lie directions are essentially the same!")
    elif S[2] / S[0] < 0.1:
        log.info("  → RANK 2: Two independent lie directions (one pair is redundant)")
    else:
        log.info("  → RANK 3: All three lie directions are independent")

    # Pairwise cosines (repeat from exp05 for completeness)
    log.info("\n  Pairwise cosines between lie directions:")
    for i in range(len(directions)):
        for j in range(i+1, len(directions)):
            cos = float(np.dot(directions[i], directions[j]))
            log.info(f"    {TYPE_LABELS[names[i]]} vs {TYPE_LABELS[names[j]]}: {cos:.4f}")

    results = {
        "singular_values": S.tolist(),
        "relative_magnitudes": (S / S[0]).tolist(),
        "cumulative_variance": cumvar.tolist(),
        "effective_rank": int(np.sum(S / S[0] > 0.1)),
        "pairwise_cosines": {},
    }
    for i in range(len(directions)):
        for j in range(i+1, len(directions)):
            results["pairwise_cosines"][f"{names[i]}_vs_{names[j]}"] = float(
                np.dot(directions[i], directions[j])
            )

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Experiment 06: Shared Deception Subspace")
    log.info("=" * 60)
    start = time.time()

    data = load_vectors()

    # Run all analyses
    X, y, type_ids, X_pca, pca = analysis_pca_visualization(data)
    shared_probe_results = analysis_shared_probe(X, y, type_ids)
    subspace_results = analysis_subspace_overlap(data)
    procrustes_results = analysis_procrustes_transfer(data)
    rank_results = analysis_lie_direction_rank(data)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    combined_acc = shared_probe_results.get("all_types_combined", {}).get("accuracy", 0)
    log.info(f"  Combined probe accuracy: {combined_acc*100:.1f}%")

    for key, val in shared_probe_results.items():
        if key.startswith("leave_out_"):
            log.info(f"  {key}: {val['accuracy']*100:.1f}%")

    effective_rank = rank_results.get("effective_rank", "?")
    log.info(f"  Lie direction effective rank: {effective_rank}/3")

    if combined_acc > 0.60:
        log.info("\n  CONCLUSION: A shared deception subspace EXISTS.")
        log.info("  Different deception types share detectable internal structure,")
        log.info("  even though their primary lie directions are orthogonal.")
    elif combined_acc > 0.55:
        log.info("\n  CONCLUSION: WEAK shared structure detected.")
        log.info("  Some commonality exists but types are mostly independent.")
    else:
        log.info("\n  CONCLUSION: No shared deception subspace found.")
        log.info("  Each deception type uses a truly independent mechanism.")

    # Save all results
    output = {
        "experiment": "06_shared_deception_subspace",
        "results": {
            "shared_probe": shared_probe_results,
            "subspace_overlap": subspace_results,
            "procrustes_transfer": procrustes_results,
            "lie_direction_rank": rank_results,
        },
        "elapsed_seconds": time.time() - start,
    }
    save_results(output, "results/exp06_shared_subspace.json")
    log.info(f"\nSaved to results/exp06_shared_subspace.json")
    log.info(f"Total time: {time.time() - start:.0f}s")

    return output


if __name__ == "__main__":
    main()
