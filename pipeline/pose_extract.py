"""
Stage 1 of the shot-classifier training pipeline: extract pose keypoints for
EVERY frame of a video, once, and cache them to disk.

This is the expensive pass (one pose inference per frame), but it only has to
run once per video. Everything downstream (dataset building, feature
experiments, retraining) then works off the cached .npz in seconds instead of
re-running inference each time.

Usage:
    python3 pipeline/pose_extract.py media/basketball_test.mov data/pose_test.npz

Output .npz contains:
    keypoints : (num_frames, 17, 2) float32  — COCO keypoint xy, full-frame coords.
                                                NaN where no person was detected.
    scores    : (num_frames, 17)    float32  — per-keypoint confidence, 0 where absent.
    fps       : scalar
"""

import sys
import numpy as np
import cv2
from ultralytics import YOLO

KEYPOINT_CONF_THRESHOLD = 0.3


def extract(video_path, out_path, model_name='yolov8n-pose.pt'):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting pose from {video_path}: {total} frames @ {fps:.1f}fps")

    all_kpts, all_scores = [], []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, verbose=False)
        kpts = np.full((17, 2), np.nan, dtype=np.float32)
        scores = np.zeros(17, dtype=np.float32)

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # If multiple people, take the one with the largest bounding box —
            # for this footage that's the shooter (closest / most prominent).
            best_i, best_area = 0, -1.0
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = float((x2 - x1) * (y2 - y1))
                    if area > best_area:
                        best_i, best_area = i, area
            if best_i < len(results[0].keypoints.xy):
                k = results[0].keypoints.xy[best_i].cpu().numpy()
                conf = results[0].keypoints.conf
                c = conf[best_i].cpu().numpy() if conf is not None else np.ones(len(k))
                n = min(17, len(k))
                kpts[:n] = k[:n]
                scores[:n] = c[:n]
                kpts[scores < KEYPOINT_CONF_THRESHOLD] = np.nan

        all_kpts.append(kpts)
        all_scores.append(scores)

        if frame_idx % 250 == 0:
            print(f"  {frame_idx}/{total} frames...")

    cap.release()
    keypoints = np.stack(all_kpts)
    scores = np.stack(all_scores)
    np.savez_compressed(out_path, keypoints=keypoints, scores=scores, fps=fps)
    detected = np.mean(~np.isnan(keypoints[:, :, 0]).all(axis=1))
    print(f"Saved {out_path}: {keypoints.shape[0]} frames, "
          f"person detected in {detected * 100:.1f}% of frames")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit("usage: pose_extract.py <video> <out.npz>")
    extract(sys.argv[1], sys.argv[2])
