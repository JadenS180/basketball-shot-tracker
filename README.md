# Basketball Shot Tracker

![Python](https://img.shields.io/badge/python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Accuracy](https://img.shields.io/badge/accuracy-28%2F29%20(96.6%25)-brightgreen)

A computer vision pipeline that watches recorded basketball footage from a single fixed camera and automatically detects each shot attempt, classifying it as a **MAKE** or **MISS**.

Built on a fine-tuned YOLOv8 model and OpenCV, running against a low-angle backyard-hoop camera — the kind of setup where the ball is small, fast, and frequently occluded, and where simple 2D geometry can be fooled by real 3D depth (a ball bouncing in front of the net looks, in a flat image, a lot like a ball dropping through it).

## Demo

![demo](assets/demo.gif)

*Live output: rim detection, the tracking regions, and the MAKE/MISS call for each shot, overlaid on the original footage.*

## Results

| Metric | Value |
|---|---|
| Shots correctly classified | **28 / 29 (96.6%)** |
| False positives | **0** |
| Make/miss errors | **0** |
| Undetected (pure ball-detection miss) | 1 |

The one miss is a case where the YOLO model never detected the ball at all during that shot's rise — see [Known limitations](#known-limitations).


## How it works

### Detection

A YOLOv8 model fine-tuned on this camera's footage detects three classes — **ball**, **rim**, and **human** — at `imgsz=1280, conf=0.05`. The high inference size matters more than it sounds: at the default `imgsz=640`, a 1920×1080 frame gets downscaled enough that the ball becomes nearly invisible to the model (measured ball-detection rate: 25.6%). At 1280, that jumps to ~96%.

### Make/miss logic — the two-region trendline method

Two tracking regions are maintained relative to the rim:

- **Up region** (blue) — spans a wide area above the rim, where the ball is tracked as it rises and arcs.
- **Down region** (green) — a narrower band directly below the rim.

When the ball is seen in the down region after having been seen in the up region, a straight line is drawn between those two points and solved for where it crosses the rim's height. If that crossing point falls within the rim's horizontal span, it's a **MAKE**; otherwise, a **MISS**.

This approach is adapted from [avishah3/AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker).

### Handling a low, fixed camera

A tripod-height camera pointed up at a residential hoop introduces problems a broadcast-angle camera doesn't have, and most of this project's iteration went into these:

- **Rattle discrimination.** A ball that rattles around the rim before dropping in (or popping back out) crosses the rim's height multiple times. Resolving on the *first* crossing scores the shot before its real outcome is known. A short confirmation window (0.25s) is held before any result is finalized — if the ball bounces back above the rim line during that window, the candidate is discarded and tracking continues rather than locking in a premature verdict.
- **Perspective / depth ambiguity.** A ball bouncing in front of the net, closer to the camera, can produce a 2D trajectory that looks like it went through the hoop even though it never did. Two independent signals catch this:
  - **Path steepness** — a real make falls close to vertically near the rim; a deflection travels sideways relative to how far it drops.
  - **Apparent size** — a ball closer to the camera reads larger in pixels. Checked at both the ball's slowest point (near its arc's apex, where motion blur doesn't distort the reading) and at the crossing point itself, against the expected size for something genuinely at the rim's depth.
- **False detections on the shooter's own body.** The ball model can occasionally mistake an arm or hand for the ball. Human detections from the same YOLO model are used to reject ball candidates that overlap a person's bounding box — but only for the wide, above-rim tracking; a layup's ball naturally overlaps the shooter near the hoop, and that's normal, not an error.
- **Clean bounce-outs.** A shot that clangs off the rim and bounces away without ever reaching the down region would otherwise never resolve. If the ball is tracked above the rim, arriving from a genuine high arc, and never subsequently appears below it within a timeout, it's scored as a MISS.
- **Anti-double-count.** A single rattling shot shouldn't score twice. A new result is only accepted once the ball has either traveled clear of the rim area or enough time has passed — the latter specifically so a fast rebound-putback from close range isn't blocked forever waiting for distance that never comes.

### Known limitations

- **Pure detection misses.** In one case in the test video, the ball is never detected by the model at all during a shot's rise (no ball-shaped box anywhere in that region of the frame) — no amount of logic tuning fixes a ball the detector never saw. Improving this further means improving the underlying ball detector (more training data, a different backbone), not the tracking logic.
- **Single fixed camera, single hoop.** The pipeline is tuned against this specific camera angle and this specific rim's apparent size. A different mounting height or distance would need the size-based checks (`APEX_OVERSIZE_MULT`, `BALL_SIZE_MAX_MULT`, etc.) recalibrated.

## Repo structure

```
pipeline/
  final_detector.py       # the shot detector — run this
  static_filter.py        # rejects persistent phantom detections (dependency)
  build_dataset.py        # builds the pose-feature dataset used for train_shot_model.py
  extract_training_frames.py
  propose_candidates.py   # semi-automated labeling workflow for shot releases
  train_ball_detector.py
  train_shot_model.py
  ball_tracker.py, shot_classifier.py, shot_pose.py, pose_extract.py,
  diagnose_features.py, tune_inference.py   # supporting tooling used during development
models/
  basketball_model.pt     # fine-tuned YOLOv8 — ball / rim / human
  hoop_model.pt            # earlier rim-only model, kept for reference
  shot_clf.joblib          # trained shot-release pose classifier
data/
  labels.txt               # labeled shot-release timestamps used for training
```

Training datasets (`models/basketball_dataset/`, `models/hoop_dataset/`), raw/output video (`media/`), and intermediate labeling artifacts (`training_frames/`, `review_clips/`) are kept locally and excluded from version control via `.gitignore` — they're large (several GB) and not needed to run the pipeline.

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python

# Place your source video at media/basketball_test.mov, then:
python3 pipeline/final_detector.py
```

Output video and a per-frame log are written to `media/final_output.mp4` and printed to stdout — redirect to a file to capture the full run.

## License

MIT — see `LICENSE`.
