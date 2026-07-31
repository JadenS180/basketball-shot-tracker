"""
Extract the frames where ball detection FAILS, for targeted annotation.

WHY NOT JUST UPLOAD RANDOM FRAMES
Annotating 500 random frames mostly teaches the model things it already knows.
The frames that actually improve it are the ones it currently gets wrong. This
finds them automatically, so your annotation time goes where it matters.

WHAT COUNTS AS A "HARD" FRAME HERE
  1. GAP frames — the ball was detected shortly before AND shortly after, but not
     in this frame. The ball is almost certainly visible; the model just missed
     it. These are the highest-value training examples.
  2. LOW-CONFIDENCE frames — detected, but barely. The model is unsure.
  3. RIM-REGION gaps — a gap frame that also happens to be near the hoop. These
     are weighted most heavily, because losing the ball at the rim is what breaks
     make/miss classification, while losing it mid-court is harmless.

OUTPUT
  training_frames/gap_XXXXX.jpg        — model saw nothing, ball probably there
  training_frames/lowconf_XXXXX.jpg    — model unsure
  training_frames/rim_XXXXX.jpg        — gap frames near the hoop (highest value)

Upload these to a Roboflow project, annotate the ball (and rim, if you want to
improve that too), then train. Because these are your camera, your lighting, and
your hoop, a few hundred of them will do more than thousands of generic
basketball images shot from a broadcast angle.

USAGE
    python3 pipeline/extract_training_frames.py media/basketball_test.mov training_frames
"""

import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

BALL_CLASS, RIM_CLASS = 0, 2
DETECT_CONF = 0.2          # what the pipeline currently uses
LOW_CONF_BAND = (0.2, 0.4)  # "detected but shaky"
GAP_LOOKAROUND = 8          # frames each side used to decide the ball was really there
MAX_PER_CATEGORY = 400      # cap so you aren't handed an unannotatable pile


def main(video_path, out_dir):
    model = YOLO('models/basketball_model.pt')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Scanning {total} frames for detection failures...")

    # Pass 1: record where the ball was and wasn't detected.
    detections = {}   # frame -> (x, y, conf)
    rim_boxes = []
    frames_cache = {}
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        best = None
        for r in model(frame, verbose=False):
            for box in r.boxes:
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                if cls_id == BALL_CLASS and conf > DETECT_CONF:
                    if best is None or conf > best[2]:
                        best = ((x1 + x2) / 2, (y1 + y2) / 2, conf)
                elif cls_id == RIM_CLASS and conf > 0.4:
                    rim_boxes.append([x1, y1, x2, y2])
        if best:
            detections[idx] = best
        if idx % 250 == 0:
            print(f"  {idx}/{total}  (ball found in {len(detections)} frames so far)")

    cap.release()
    detected_pct = 100.0 * len(detections) / max(idx, 1)
    print(f"\nBall detected in {len(detections)}/{idx} frames ({detected_pct:.1f}%)")

    rim = np.median(np.array(rim_boxes), axis=0) if rim_boxes else None
    if rim is not None:
        rim_cx, rim_cy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
        rim_w = rim[2] - rim[0]
        print(f"Rim at ({rim_cx:.0f},{rim_cy:.0f}), width {rim_w:.0f}px")
    else:
        rim_cx = rim_cy = rim_w = None

    # Decide which frames are worth annotating.
    gaps, lowconf, rim_gaps = [], [], []
    for f in range(1, idx + 1):
        if f in detections:
            if LOW_CONF_BAND[0] < detections[f][2] < LOW_CONF_BAND[1]:
                lowconf.append(f)
            continue
        # Not detected. Was it detected shortly before AND after? Then it was there.
        before = [d for d in range(max(1, f - GAP_LOOKAROUND), f) if d in detections]
        after = [d for d in range(f + 1, min(idx, f + GAP_LOOKAROUND) + 1) if d in detections]
        if not (before and after):
            continue
        # Interpolate roughly where the ball should have been.
        bx, by = detections[before[-1]][0], detections[before[-1]][1]
        ax, ay = detections[after[0]][0], detections[after[0]][1]
        mx, my = (bx + ax) / 2, (by + ay) / 2
        near_rim = (rim_cx is not None and abs(mx - rim_cx) < rim_w * 3
                    and abs(my - rim_cy) < rim_w * 4)
        (rim_gaps if near_rim else gaps).append(f)

    print(f"\nFound:")
    print(f"  {len(rim_gaps):5d} gap frames NEAR THE RIM   (highest training value)")
    print(f"  {len(gaps):5d} gap frames elsewhere")
    print(f"  {len(lowconf):5d} low-confidence frames")

    # Pass 2: write out the selected frames.
    os.makedirs(out_dir, exist_ok=True)
    wanted = {}
    for f in rim_gaps[:MAX_PER_CATEGORY]:
        wanted[f] = 'rim'
    for f in gaps[:MAX_PER_CATEGORY]:
        wanted.setdefault(f, 'gap')
    for f in lowconf[:MAX_PER_CATEGORY]:
        wanted.setdefault(f, 'lowconf')

    cap = cv2.VideoCapture(video_path)
    idx = written = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in wanted:
            cv2.imwrite(os.path.join(out_dir, f"{wanted[idx]}_{idx:05d}.jpg"), frame)
            written += 1
    cap.release()

    print(f"\nWrote {written} frames to {out_dir}/")
    print("\nNEXT STEPS:")
    print(f"  1. Create a Roboflow project (Object Detection), upload {out_dir}/")
    print("  2. Annotate the ball in each — prioritise the rim_*.jpg files if short on time")
    print("  3. Generate a version with augmentation (flip, brightness, blur — blur")
    print("     especially, since motion blur is a big part of why the ball is missed)")
    print("  4. Export as YOLOv8, then fine-tune with train_ball_detector.py")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit("usage: extract_training_frames.py <video> <out_dir>")
    main(sys.argv[1], sys.argv[2])
