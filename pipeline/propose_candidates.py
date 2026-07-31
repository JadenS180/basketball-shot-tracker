"""
Semi-automatic labeling. You should not be hand-hunting timestamps.

WORKFLOW:
  1. This script scans the cached pose data with a deliberately LOOSE,
     high-recall motion detector and proposes every moment that could
     plausibly be a shot release. It over-proposes on purpose — missing a real
     shot here is unrecoverable, whereas a junk candidate costs you one click.

  2. It cuts a ~1.5s clip around each candidate into review_clips/, named with
     its index and timestamp.

  3. YOU: open review_clips/ in Finder, press space to Quick Look, arrow
     through them. DELETE any clip that isn't a real shot release. That's it —
     no typing, no timestamps.

  4. Run  finalize_labels.py  (bottom of this file, or via --finalize) to
     rebuild labels.txt from whichever clips survived.

Labels only need to be accurate to ~±0.3s, since the feature window is ~0.4s
wide and each label is jittered ±2 frames during dataset building.

Usage:
    python3 pipeline/propose_candidates.py data/pose_test.npz media/basketball_test.mov review_clips
    # ...delete non-shot clips in Finder...
    python3 pipeline/propose_candidates.py --finalize review_clips data/labels.txt
"""

import os
import re
import sys
import numpy as np
import cv2

L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

CLIP_PAD_S = 0.75           # clip extends this far each side of the candidate
MIN_SEPARATION_S = 0.8      # merge candidates closer together than this
# Loose thresholds — deliberately permissive. Over-proposing is cheap here.
TORSO_RISE_THRESHOLD = 0.35  # torso lengths per second
WRIST_RISE_THRESHOLD = 0.5   # torso lengths per second


def _centroid(kpts, idxs):
    pts = [kpts[i] for i in idxs if not np.any(np.isnan(kpts[i]))]
    return np.mean(pts, axis=0) if pts else np.array([np.nan, np.nan])


def propose(pose_path, video_path, out_dir):
    data = np.load(pose_path)
    keypoints, fps = data['keypoints'], float(data['fps'])
    n = len(keypoints)

    # Per-frame normalized torso and wrist heights.
    torso_y = np.full(n, np.nan)
    wrist_y = np.full(n, np.nan)
    for i in range(n):
        k = keypoints[i]
        sh, hip = _centroid(k, [L_SHOULDER, R_SHOULDER]), _centroid(k, [L_HIP, R_HIP])
        if np.any(np.isnan(sh)) or np.any(np.isnan(hip)):
            continue
        tl = np.linalg.norm(sh - hip)
        if tl < 1e-3:
            continue
        torso_y[i] = ((sh[1] + hip[1]) / 2) / tl
        wr = [k[w][1] for w in (R_WRIST, L_WRIST) if not np.any(np.isnan(k[w]))]
        if wr:
            wrist_y[i] = min(wr) / tl

    # Rise rate over a short lookback, in torso-lengths/sec (scale invariant).
    # IMPORTANT: we require the rise to be SUSTAINED over several consecutive
    # frames rather than trusting any single frame. Pose estimation produces
    # isolated spikes (a keypoint jumps for one frame) that can easily exceed
    # the magnitude of a real shot motion; requiring persistence rejects those.
    # The candidate is anchored at the END of each sustained run, since that's
    # nearest the actual release.
    lookback = max(2, int(fps * 0.2))
    rising = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        t_rise = w_rise = 0.0
        dt = lookback / fps
        if not np.isnan(torso_y[i]) and not np.isnan(torso_y[i - lookback]):
            t_rise = (torso_y[i - lookback] - torso_y[i]) / dt
        if not np.isnan(wrist_y[i]) and not np.isnan(wrist_y[i - lookback]):
            w_rise = (wrist_y[i - lookback] - wrist_y[i]) / dt
        rising[i] = (t_rise > TORSO_RISE_THRESHOLD or w_rise > WRIST_RISE_THRESHOLD)

    min_run = max(3, int(fps * 0.12))
    merged, run_start = [], None
    for i in range(n):
        if rising[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_run:
                merged.append((i - 1, float(i - run_start)))  # anchor at end of run
            run_start = None
    if run_start is not None and (n - run_start) >= min_run:
        merged.append((n - 1, float(n - run_start)))

    # Collapse candidates that are still too close together.
    min_sep = int(fps * MIN_SEPARATION_S)
    collapsed = []
    for idx, strength in merged:
        if collapsed and idx - collapsed[-1][0] < min_sep:
            if strength > collapsed[-1][1]:
                collapsed[-1] = (idx, strength)
        else:
            collapsed.append((idx, strength))
    merged = collapsed

    print(f"Proposed {len(merged)} candidate releases (loose/high-recall on purpose).")

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pad = int(fps * CLIP_PAD_S)

    for ci, (idx, strength) in enumerate(merged):
        t = idx / fps
        start, end = max(0, idx - pad), min(n - 1, idx + pad)
        name = f"cand_{ci:03d}_t{t:07.2f}.mp4"
        out = cv2.VideoWriter(os.path.join(out_dir, name), fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for f in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # Mark the proposed release frame so you can see what's being labeled.
            if f == idx:
                cv2.putText(frame, "<< RELEASE?", (60, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            cv2.putText(frame, f"cand {ci}  t={t:.2f}s", (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            out.write(frame)
        out.release()
        if (ci + 1) % 20 == 0:
            print(f"  wrote {ci + 1}/{len(merged)} clips...")

    cap.release()
    print(f"\nWrote {len(merged)} clips to {out_dir}/")
    print("NEXT: open that folder, Quick Look through the clips, and DELETE any that")
    print("      are not a real shot release (dribbles, passes, rebounds, walking).")
    print(f"THEN: python3 pipeline/propose_candidates.py --finalize {out_dir} data/labels.txt")


def finalize(clip_dir, labels_path):
    times = []
    for fn in sorted(os.listdir(clip_dir)):
        m = re.match(r"cand_\d+_t(\d+\.\d+)\.mp4$", fn)
        if m:
            times.append(float(m.group(1)))
    times.sort()
    os.makedirs(os.path.dirname(labels_path) or '.', exist_ok=True)
    with open(labels_path, 'w') as f:
        f.write("# Auto-generated from surviving review clips. One release time (s) per line.\n")
        for t in times:
            f.write(f"{t:.2f}\n")
    print(f"Wrote {len(times)} labels to {labels_path}")
    if len(times) < 20:
        print("NOTE: fewer than 20 positives is thin for training — if your real shot")
        print("      count is higher, check whether good clips got deleted by mistake.")


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == '--finalize':
        if len(sys.argv) < 4:
            raise SystemExit("usage: propose_candidates.py --finalize <clip_dir> <labels.txt>")
        finalize(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) < 4:
            raise SystemExit("usage: propose_candidates.py <pose.npz> <video> <clip_dir>")
        propose(sys.argv[1], sys.argv[2], sys.argv[3])
