"""
Diagnose feature health before trusting a trained model.

Two failure modes this catches:
  1. DEAD FEATURES — keypoints that were usually missing (e.g. legs out of
     frame on a low camera) get filled with 0.0, becoming near-constant columns
     the model cannot learn from. They look "present" but carry no information.
  2. WEAK FEATURES — present and varying, but with no ability to separate
     shots from non-shots on their own.

If most features are dead, the model is effectively relying on one or two
signals, which is fragile regardless of what the headline accuracy says.

Usage:
    python3 pipeline/diagnose_features.py data/dataset.npz
"""

import sys
import numpy as np
from sklearn.metrics import roc_auc_score


def diagnose(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    X, y = data['X'], data['y']
    names = [str(n) for n in data['feature_names']]

    print(f"{X.shape[0]} samples, {X.shape[1]} features, {int(y.sum())} positive\n")
    print(f"{'feature':<34}{'%zero':>8}{'std':>10}{'AUC':>8}  verdict")
    print("-" * 74)

    dead, weak, useful = [], [], []
    for i, name in enumerate(names):
        col = X[:, i].astype(np.float64)
        pct_zero = 100.0 * np.mean(col == 0.0)
        std = float(np.std(col))
        try:
            auc = roc_auc_score(y, col)
            auc = max(auc, 1 - auc)  # direction-agnostic separating power
        except ValueError:
            auc = 0.5

        if pct_zero > 50 or std < 1e-6:
            verdict = "DEAD (mostly missing/constant)"
            dead.append(name)
        elif auc < 0.6:
            verdict = "weak"
            weak.append(name)
        else:
            verdict = "useful"
            useful.append(name)
        print(f"{name:<34}{pct_zero:>7.1f}%{std:>10.3f}{auc:>8.3f}  {verdict}")

    print("-" * 74)
    print(f"\nDEAD:   {len(dead):>2} features — {', '.join(dead) if dead else 'none'}")
    print(f"WEAK:   {len(weak):>2} features")
    print(f"USEFUL: {len(useful):>2} features — {', '.join(useful) if useful else 'none'}")

    if dead:
        print("\nDead features mean those keypoints were usually not visible. On a low,")
        print("close camera this typically means legs/ankles are out of frame, killing")
        print("all the knee-based shooting-mechanics features. Options:")
        print("  - reframe the camera to keep the full body in shot, then re-extract, or")
        print("  - drop those features and rely on upper-body cues only.")
    if len(useful) <= 2:
        print("\nWARNING: the model is leaning on very few signals. Headline accuracy")
        print("         will not survive a different camera angle or session.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("usage: diagnose_features.py <dataset.npz>")
    diagnose(sys.argv[1])
