"""
Pretrained-pose-based shooting-motion scoring.

No training involved — this loads Ultralytics' pretrained YOLOv8-pose model
(COCO 17-keypoint format) and scores a few simple, depth-invariant geometric
cues associated with a jump-shot release:

  1. wrist raised above the shoulder
  2. shooting arm meaningfully extended (shoulder-elbow-wrist angle)
  3. wrist raised above the head/nose

These are checked against the PLAYER's own body proportions, not the rim or
the frame — which is exactly what makes this useful where rim/frame-relative
pixel thresholds fall apart (a camera that's close, low, and wide-angle has
a huge depth range in one frame, so anything scaled to the rim's distance
doesn't hold for something happening at a different distance from the camera).

This is intentionally NOT a trained shot classifier — it's hand-written
geometry over pretrained keypoints. Treat the returned score as one
corroborating signal alongside the physics checks, not a standalone verdict:
pose detection can fail (partial occlusion, player mostly out of frame), and
in those cases the physics-only checks should still be allowed to carry a
shot through on their own.
"""

import numpy as np
from ultralytics import YOLO

pose_model = YOLO('yolov8n-pose.pt')  # auto-downloads on first use

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

KEYPOINT_CONF_THRESHOLD = 0.3


def _angle(a, b, c):
    """Angle at point b, between vectors b->a and b->c, in degrees."""
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def shooting_pose_score(frame, human_box, pad_frac=0.15):
    """
    Run pose estimation on a crop around the given human box and score
    shooting-motion cues. Returns None if pose estimation fails to find a
    usable person (caller should treat that as "no signal", not "not a shot").
    """
    x1, y1, x2, y2 = human_box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    px1 = max(0, int(x1 - w * pad_frac))
    py1 = max(0, int(y1 - h * pad_frac))
    px2 = int(x2 + w * pad_frac)
    py2 = int(y2 + h * pad_frac)
    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return None

    results = pose_model(crop, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None

    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2), crop-local coords
    conf_tensor = results[0].keypoints.conf
    kconf = conf_tensor[0].cpu().numpy() if conf_tensor is not None else np.ones(17)
    if kpts.shape[0] < 17:
        return None
    kpts[:, 0] += px1  # back to full-frame coordinates, for consistency/debugging
    kpts[:, 1] += py1

    def valid(i):
        return kconf[i] > KEYPOINT_CONF_THRESHOLD

    checks = {}

    wrist_raised = False
    for side, sh, wr in [("R", R_SHOULDER, R_WRIST), ("L", L_SHOULDER, L_WRIST)]:
        if valid(sh) and valid(wr) and kpts[wr][1] < kpts[sh][1] - 0.1 * h:
            wrist_raised = True
    checks["wrist_above_shoulder"] = wrist_raised

    elbow_extended = False
    for side, sh, el, wr in [("R", R_SHOULDER, R_ELBOW, R_WRIST), ("L", L_SHOULDER, L_ELBOW, L_WRIST)]:
        if valid(sh) and valid(el) and valid(wr):
            if _angle(kpts[sh], kpts[el], kpts[wr]) > 140:
                elbow_extended = True
    checks["elbow_extended"] = elbow_extended

    wrist_above_head = False
    if valid(NOSE):
        for wr in (R_WRIST, L_WRIST):
            if valid(wr) and kpts[wr][1] < kpts[NOSE][1]:
                wrist_above_head = True
    checks["wrist_above_head"] = wrist_above_head

    # "wrist above shoulder" is nearly always true for anyone just holding/dribbling a
    # ball — it barely discriminates anything, so it's weighted low. Elbow extension and
    # wrist-above-head are the checks that actually distinguish a release from a hold.
    weights = {"wrist_above_shoulder": 0.5, "elbow_extended": 1.5, "wrist_above_head": 1.5}
    weighted_score = sum(weights[k] for k, v in checks.items() if v)
    max_score = sum(weights.values())
    strong_signal = elbow_extended or wrist_above_head  # callers can require this directly
    release_like = elbow_extended and wrist_above_head  # stricter: needed to trigger a shot
                                                          # with NO ball evidence at all

    # Highest (most raised, smallest y) confidently-tracked wrist — used as a position
    # proxy for the ball when the ball itself isn't detected at release.
    wrist_pos = None
    for wr in (R_WRIST, L_WRIST):
        if valid(wr) and (wrist_pos is None or kpts[wr][1] < wrist_pos[1]):
            wrist_pos = (float(kpts[wr][0]), float(kpts[wr][1]))

    # Torso centroid (hips + shoulders). Critically, these keypoints stay visible when
    # the player's BACK is turned to the camera — unlike wrists, which are frequently
    # occluded by the body in exactly that situation. A jump shot's explosive upward
    # torso movement is a strong, view-angle-agnostic release cue.
    torso_pts = [i for i in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER) if valid(i)]
    torso_pos = None
    if len(torso_pts) >= 2:
        torso_pos = (float(np.mean([kpts[i][0] for i in torso_pts])),
                     float(np.mean([kpts[i][1] for i in torso_pts])))

    return {"score": weighted_score, "total": max_score, "fraction": weighted_score / max_score,
            "checks": checks, "strong_signal": strong_signal, "release_like": release_like,
            "wrist_pos": wrist_pos, "torso_pos": torso_pos, "keypoints": kpts}
