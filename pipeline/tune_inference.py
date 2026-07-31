"""
Find the best inference settings for ball detection — NO retraining, NO annotation.

THE THING THAT MATTERS MOST HERE
Ultralytics resizes every frame to imgsz=640 by default. This video is 1920x1080,
so the ball — already small — is shrunk to a third of its size before the model
sees it. Small-object detection is extremely sensitive to input resolution, so
simply running inference at native resolution often produces a large jump in
detection rate at zero cost.

Confidence threshold is the second free lever: detections the model already makes
but discards below the cutoff.

This sweeps both and reports the ball-detection rate for each combination, so you
can pick the best settings and just use them.

USAGE
    python3 pipeline/tune_inference.py media/basketball_test.mov
"""

import sys

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = 'models/basketball_model.pt'
BALL_CLASS, RIM_CLASS = 0, 2

IMAGE_SIZES = [640, 960, 1280, 1920]
CONF_LEVELS = [0.05, 0.10, 0.20]
N_SAMPLES = 150          # sampled frames per combination


def main(video_path):
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sorted(set(np.linspace(1, total - 1, N_SAMPLES).astype(int).tolist()))

    # Cache the sampled frames once so every combination sees identical input.
    print(f"Loading {len(idxs)} sample frames from {total}...")
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    print(f"Loaded {len(frames)} frames\n")

    # Locate the rim once (at high res, so it's reliable) to measure the
    # near-rim detection rate — the number that actually decides shot outcomes.
    rim_boxes = []
    for f in frames[:40]:
        for r in model(f, imgsz=1280, conf=0.4, verbose=False):
            for b in r.boxes:
                if int(b.cls[0]) == RIM_CLASS:
                    rim_boxes.append(list(map(float, b.xyxy[0])))
    if rim_boxes:
        rim = np.median(np.array(rim_boxes), axis=0)
        rim_cx, rim_cy = (rim[0] + rim[2]) / 2, (rim[1] + rim[3]) / 2
        rim_w = rim[2] - rim[0]
        print(f"Rim at ({rim_cx:.0f},{rim_cy:.0f}) width {rim_w:.0f}px\n")
    else:
        rim_cx = rim_cy = rim_w = None

    print(f"{'imgsz':>7}{'conf':>7}{'ball found':>13}{'rate':>8}{'near rim':>11}")
    print("-" * 48)

    results = []
    for imgsz in IMAGE_SIZES:
        for conf in CONF_LEVELS:
            hits = rim_hits = 0
            for f in frames:
                best = None
                for r in model(f, imgsz=imgsz, conf=conf, verbose=False):
                    for b in r.boxes:
                        if int(b.cls[0]) == BALL_CLASS:
                            c = float(b.conf[0])
                            x1, y1, x2, y2 = map(float, b.xyxy[0])
                            if best is None or c > best[2]:
                                best = ((x1 + x2) / 2, (y1 + y2) / 2, c)
                if best:
                    hits += 1
                    if rim_cx is not None and abs(best[0] - rim_cx) < rim_w * 3 \
                            and abs(best[1] - rim_cy) < rim_w * 4:
                        rim_hits += 1
            rate = 100.0 * hits / max(len(frames), 1)
            results.append((imgsz, conf, hits, rate, rim_hits))
            print(f"{imgsz:>7}{conf:>7.2f}{hits:>13}{rate:>7.1f}%{rim_hits:>11}")

    print("-" * 48)
    best = max(results, key=lambda r: r[3])
    print(f"\nBEST: imgsz={best[0]} conf={best[1]:.2f} -> {best[3]:.1f}% detection rate")
    baseline = next((r for r in results if r[0] == 640 and r[1] == 0.20), None)
    if baseline:
        print(f"Baseline (imgsz=640 conf=0.20, what you're running now): {baseline[3]:.1f}%")
        if best[3] > baseline[3]:
            print(f"Improvement: +{best[3] - baseline[3]:.1f} percentage points, free.")

    print("\nTO APPLY: in simple_detector.py / shot_detector.py, change every")
    print(f"    model(frame, verbose=False)")
    print("to")
    print(f"    model(frame, imgsz={best[0]}, conf={best[1]:.2f}, verbose=False)")
    print("\nNOTE: higher imgsz is slower per frame. If the full run gets too slow,")
    print("      step down one size and check whether the rate holds up.")
    print("Also watch for FALSE positives at low confidence — check the output video,")
    print("since a higher raw detection rate isn't automatically better if some of")
    print("those detections aren't the ball.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("usage: tune_inference.py <video>")
    main(sys.argv[1])
