"""
Stage 3: train a shot-release classifier on the engineered pose features.

MODEL CHOICE: gradient-boosted trees, not a neural net. With a few dozen
positive examples, a deep sequence model would memorize the training clips and
generalize poorly. A small tree ensemble on ~30 hand-designed, scale-invariant
features is the right capacity for this data scale, trains in seconds, and —
usefully — tells you which features actually carry signal.

The cross-validation here is GROUPED BY SHOT: the jittered copies of a single
release (offsets -2/0/+2) always land in the same fold. Without that, the model
would be scored on near-duplicates of its own training data and the reported
accuracy would be meaningless.

Usage:
    python3 pipeline/train_shot_model.py data/dataset.npz models/shot_clf.joblib
"""

import sys
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

JITTER_COPIES = 3  # must match the number of offsets used in build_dataset.py


def train(dataset_path, model_path):
    data = np.load(dataset_path, allow_pickle=True)
    X, y = data['X'], data['y']
    names = list(data['feature_names'])
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(y.sum())} positive")

    # Group jittered copies of the same shot together so CV can't cheat.
    groups = np.empty(len(y), dtype=int)
    pos_idx = np.where(y == 1)[0]
    for n, i in enumerate(pos_idx):
        groups[i] = n // JITTER_COPIES
    neg_start = groups[pos_idx].max() + 1 if len(pos_idx) else 0
    groups[y == 0] = neg_start + np.arange((y == 0).sum())

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=4,
        min_samples_leaf=5, l2_regularization=1.0, random_state=0,
    )

    n_splits = min(5, max(2, int(y.sum()) // JITTER_COPIES))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]

    print(f"\n--- Cross-validated performance ({n_splits}-fold, grouped by shot) ---")
    print(classification_report(y, (proba >= 0.5).astype(int),
                                target_names=['no-shot', 'shot'], zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, (proba >= 0.5).astype(int)))

    # A release detector should lean toward RECALL: a missed release means the
    # shot never enters the pipeline at all, whereas a false trigger still has
    # to survive the downstream physics/arc validation before it counts.
    prec, rec, thr = precision_recall_curve(y, proba)
    print("\nThreshold options (pick based on how much downstream filtering you trust):")
    for target_recall in (0.95, 0.90, 0.80):
        ok = np.where(rec[:-1] >= target_recall)[0]
        if len(ok):
            i = ok[-1]
            print(f"  recall={rec[i]:.2f}  precision={prec[i]:.2f}  threshold={thr[i]:.3f}")

    clf.fit(X, y)
    joblib.dump({'model': clf, 'feature_names': names}, model_path)
    print(f"\nSaved model to {model_path}")

    # Permutation importance tells us which cues actually mattered — useful for
    # sanity-checking that the model learned shooting mechanics and not an artifact.
    from sklearn.inspection import permutation_importance
    r = permutation_importance(clf, X, y, n_repeats=10, random_state=0, scoring='average_precision')
    order = np.argsort(r.importances_mean)[::-1][:10]
    print("\nTop 10 most informative features:")
    for i in order:
        print(f"  {names[i]:<32} {r.importances_mean[i]:+.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit("usage: train_shot_model.py <dataset.npz> <model.joblib>")
    train(sys.argv[1], sys.argv[2])
