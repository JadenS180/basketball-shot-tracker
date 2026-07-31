"""
Fine-tune the ball detector on your Roboflow-exported dataset.

IMPORTANT: FINE-TUNE, DON'T TRAIN FROM SCRATCH.
This starts from your existing basketball_model.pt rather than random weights.
Your current model already knows what a basketball looks like in general; the
goal is to teach it your specific camera, lighting, motion blur, and hoop. That
takes far less data and far fewer epochs than starting over, and it won't
forget what it already does well.

EXPECTED DATASET LAYOUT (this is what Roboflow's YOLOv8 export gives you):
    dataset/
      data.yaml
      train/images/  train/labels/
      valid/images/  valid/labels/

CLASS ORDER MATTERS. The pipeline assumes ball=0, human=1, rim=2. Open the
exported data.yaml and confirm the `names:` list matches that order — if
Roboflow exported them alphabetically or with different names, either reorder
them in data.yaml before training, or update the class constants in
shot_detector.py / simple_detector.py to match. A silent class-order mismatch
will make the pipeline behave bizarrely and is annoying to diagnose.

USAGE
    python3 pipeline/train_ball_detector.py dataset/data.yaml models/ball_v2.pt
"""

import sys

from ultralytics import YOLO

BASE_MODEL = 'models/basketball_model.pt'   # fine-tune from your existing weights
EPOCHS = 80              # fine-tuning converges much sooner than training fresh
IMAGE_SIZE = 1280        # larger than the 640 default: the ball is SMALL in your
                         # wide-angle frames, and small-object recall is very
                         # sensitive to input resolution. This matters more than
                         # epoch count for this particular problem.
BATCH = 8                # lower if you hit out-of-memory on a laptop GPU/CPU
PATIENCE = 20            # early stop if validation stops improving


def main(data_yaml, out_path):
    print(f"Fine-tuning {BASE_MODEL} on {data_yaml}")
    print(f"  epochs={EPOCHS}  imgsz={IMAGE_SIZE}  batch={BATCH}")
    print("  (imgsz is deliberately high — the ball is small in these frames)\n")

    model = YOLO(BASE_MODEL)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        # Augmentation aimed at the actual failure modes: motion blur when the
        # ball is fast, and brightness swings from outdoor light.
        hsv_v=0.5,           # brightness variation
        degrees=5,           # slight rotation
        scale=0.5,           # scale variation — ball size varies hugely with depth
        fliplr=0.5,
        mosaic=1.0,
        project='runs',
        name='ball_finetune',
        exist_ok=True,
    )

    metrics = model.val()
    print("\nValidation results:")
    try:
        print(f"  mAP50    : {metrics.box.map50:.3f}")
        print(f"  mAP50-95 : {metrics.box.map:.3f}")
        print(f"  precision: {metrics.box.mp:.3f}")
        print(f"  recall   : {metrics.box.mr:.3f}   <-- the number that matters most here")
    except Exception:
        print("  (couldn't parse metrics object; see the printed table above)")

    model.save(out_path)
    print(f"\nSaved to {out_path}")
    print("\nNOW VERIFY IT ACTUALLY HELPS — don't assume:")
    print("  1. Point simple_detector.py at the new weights and re-run")
    print("  2. Compare detection rate and make/miss accuracy against your 16/29 ground truth")
    print("  Validation mAP can improve while real-world pipeline results don't, because")
    print("  the val set is drawn from the same frames you annotated.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit("usage: train_ball_detector.py <data.yaml> <out_model.pt>")
    main(sys.argv[1], sys.argv[2])
