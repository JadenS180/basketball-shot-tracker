"""
Stage 2: turn cached pose keypoints into a labeled, learnable dataset.

THE KEY DESIGN DECISION IS IN THE FEATURES.

Every heuristic we tried before this broke on the same thing: raw pixel
measurements aren't comparable across the frame, because a wide-angle camera
mounted low and close means the same real-world motion produces wildly
different pixel numbers depending on how far the player is from the lens.

So every feature here is normalized by the player's OWN torso length
(shoulder-centroid to hip-centroid distance) and expressed in torso-lengths
rather than pixels. A shot released 30 feet away and one released 5 feet away
produce nearly identical feature values. Angles (elbow, knee) are inherently
scale-free and need no normalization at all.

That's the whole reason this should generalize where the hand-tuned thresholds
did not.

Usage:
    python3 pipeline/build_dataset.py data/pose_test.npz data/labels.txt data/dataset.npz

labels.txt: one release TIMESTAMP per line, in seconds (e.g. "5.2"), or
            "MM:SS.s" (e.g. "1:18.4"). Comments with # are ignored.
"""

import sys
import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

WINDOW_FRAMES = 12          # ~0.4s at 30fps — enough to capture the release motion
NEGATIVE_EXCLUSION_FRAMES = 20  # don't sample negatives this close to a real release


def _angle(a, b, c):
    """Angle at joint b in degrees. NaN-safe: returns NaN if any point missing."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def _centroid(kpts, idxs):
    pts = [kpts[i] for i in idxs if not np.any(np.isnan(kpts[i]))]
    if not pts:
        return np.array([np.nan, np.nan])
    return np.mean(pts, axis=0)


def frame_descriptors(kpts):
    """Scale-invariant per-frame descriptors. All distances in TORSO LENGTHS."""
    shoulder_c = _centroid(kpts, [L_SHOULDER, R_SHOULDER])
    hip_c = _centroid(kpts, [L_HIP, R_HIP])
    if np.any(np.isnan(shoulder_c)) or np.any(np.isnan(hip_c)):
        return None

    torso_len = np.linalg.norm(shoulder_c - hip_c)
    if torso_len < 1e-3:
        return None

    d = {}
    # Wrist height above shoulder, in torso lengths (image y grows downward,
    # so we negate to make "higher" positive).
    for name, wr in (("r_wrist", R_WRIST), ("l_wrist", L_WRIST)):
        if not np.any(np.isnan(kpts[wr])):
            d[f"{name}_above_shoulder"] = float((shoulder_c[1] - kpts[wr][1]) / torso_len)
        else:
            d[f"{name}_above_shoulder"] = np.nan

    # Joint angles — inherently scale-free, and the clearest shooting-form cues.
    d["r_elbow_angle"] = _angle(kpts[R_SHOULDER], kpts[R_ELBOW], kpts[R_WRIST])
    d["l_elbow_angle"] = _angle(kpts[L_SHOULDER], kpts[L_ELBOW], kpts[L_WRIST])
    d["r_knee_angle"] = _angle(kpts[R_HIP], kpts[R_KNEE], kpts[R_ANKLE])
    d["l_knee_angle"] = _angle(kpts[L_HIP], kpts[L_KNEE], kpts[L_ANKLE])

    # Absolute positions (used only for frame-to-frame deltas, normalized below).
    d["_hip_y"] = float(hip_c[1] / torso_len)
    d["_hip_x"] = float(hip_c[0] / torso_len)
    d["_shoulder_y"] = float(shoulder_c[1] / torso_len)
    d["_torso_len"] = float(torso_len)

    # Wrist above head — visible-from-most-angles release cue.
    if not np.any(np.isnan(kpts[NOSE])):
        wr_ys = [kpts[w][1] for w in (R_WRIST, L_WRIST) if not np.any(np.isnan(kpts[w]))]
        d["wrist_above_head"] = float(kpts[NOSE][1] - min(wr_ys)) / torso_len if wr_ys else np.nan
    else:
        d["wrist_above_head"] = np.nan
    return d


DESCRIPTOR_KEYS = ["r_wrist_above_shoulder", "l_wrist_above_shoulder",
                   "r_elbow_angle", "l_elbow_angle",
                   "r_knee_angle", "l_knee_angle", "wrist_above_head"]


def window_features(keypoints, center_idx, fps):
    """
    Build one feature vector from a window of frames ending at center_idx.
    Captures both the pose itself AND how it's changing — a jump shot is
    defined by the motion (knees extend, arms extend, body rises), not any
    single static frame.
    """
    start = center_idx - WINDOW_FRAMES + 1
    if start < 0 or center_idx >= len(keypoints):
        return None

    descs = [frame_descriptors(keypoints[i]) for i in range(start, center_idx + 1)]
    valid = [d for d in descs if d is not None]
    if len(valid) < WINDOW_FRAMES * 0.5:  # need at least half the window usable
        return None

    feats, names = [], []

    # Static/aggregate descriptors over the window.
    for key in DESCRIPTOR_KEYS:
        vals = np.array([d[key] for d in valid], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            feats.extend([0.0, 0.0, 0.0]); names.extend([f"{key}_mean", f"{key}_max", f"{key}_range"])
            continue
        feats.extend([float(np.mean(vals)), float(np.max(vals)), float(np.ptp(vals))])
        names.extend([f"{key}_mean", f"{key}_max", f"{key}_range"])

    # Motion features: how fast the body rose over the window (torso lengths/sec).
    hip_ys = np.array([d["_hip_y"] for d in valid], dtype=np.float64)
    sh_ys = np.array([d["_shoulder_y"] for d in valid], dtype=np.float64)
    dt = len(valid) / fps
    feats.append(float((hip_ys[0] - hip_ys[-1]) / dt))     # + = hips rose
    feats.append(float((sh_ys[0] - sh_ys[-1]) / dt))       # + = shoulders rose
    feats.append(float(np.ptp(hip_ys)))                    # total vertical hip travel
    names.extend(["hip_rise_rate", "shoulder_rise_rate", "hip_vertical_range"])

    # Knee extension rate — the "load then explode" signature of a jump shot.
    knee = np.array([np.nanmean([d["r_knee_angle"], d["l_knee_angle"]]) for d in valid],
                    dtype=np.float64)
    knee = knee[~np.isnan(knee)]
    if len(knee) >= 2:
        feats.extend([float((knee[-1] - knee[0]) / dt), float(np.ptp(knee))])
    else:
        feats.extend([0.0, 0.0])
    names.extend(["knee_extension_rate", "knee_angle_range"])

    # Elbow extension rate — the release itself.
    elbow = np.array([np.nanmean([d["r_elbow_angle"], d["l_elbow_angle"]]) for d in valid],
                     dtype=np.float64)
    elbow = elbow[~np.isnan(elbow)]
    if len(elbow) >= 2:
        feats.extend([float((elbow[-1] - elbow[0]) / dt), float(np.ptp(elbow))])
    else:
        feats.extend([0.0, 0.0])
    names.extend(["elbow_extension_rate", "elbow_angle_range"])

    return np.array(feats, dtype=np.float32), names


def parse_timestamp(line):
    line = line.split('#')[0].strip()
    if not line:
        return None
    if ':' in line:
        mins, secs = line.split(':')
        return float(mins) * 60 + float(secs)
    return float(line)


def build(pose_path, labels_path, out_path):
    data = np.load(pose_path)
    keypoints, fps = data['keypoints'], float(data['fps'])

    with open(labels_path) as f:
        times = [t for t in (parse_timestamp(l) for l in f) if t is not None]
    release_frames = sorted(int(round(t * fps)) for t in times)
    print(f"Loaded {len(release_frames)} labeled releases from {labels_path}")

    X, y, names = [], [], None

    # Positives: windows ENDING at (and just around) each labeled release, since
    # the discriminative motion is the run-up to the release, not what follows.
    for rf in release_frames:
        for offset in (-2, 0, 2):  # slight jitter = cheap augmentation
            res = window_features(keypoints, rf + offset, fps)
            if res is not None:
                feat, names = res
                X.append(feat); y.append(1)

    # Negatives: everything comfortably away from any labeled release.
    exclusion = set()
    for rf in release_frames:
        exclusion.update(range(rf - NEGATIVE_EXCLUSION_FRAMES, rf + NEGATIVE_EXCLUSION_FRAMES + 1))
    rng = np.random.default_rng(0)
    candidates = [i for i in range(WINDOW_FRAMES, len(keypoints)) if i not in exclusion]
    rng.shuffle(candidates)
    target_negatives = min(len(candidates), max(200, len(X) * 8))
    for i in candidates:
        if sum(1 for v in y if v == 0) >= target_negatives:
            break
        res = window_features(keypoints, i, fps)
        if res is not None:
            feat, names = res
            X.append(feat); y.append(0)

    X = np.stack(X); y = np.array(y)
    np.savez_compressed(out_path, X=X, y=y, feature_names=np.array(names))
    print(f"Saved {out_path}: {X.shape[0]} samples ({int(y.sum())} positive, "
          f"{int((y == 0).sum())} negative), {X.shape[1]} features each")
    if y.sum() < 20:
        print("WARNING: very few positive examples. Expect weak generalization — "
              "labeling a second video would help a lot.")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit("usage: build_dataset.py <pose.npz> <labels.txt> <out.npz>")
    build(sys.argv[1], sys.argv[2], sys.argv[3])
