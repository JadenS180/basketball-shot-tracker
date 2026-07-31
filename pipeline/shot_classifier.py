"""
Live inference wrapper around the trained shot-release classifier.

Maintains a rolling buffer of pose keypoints and, on demand, builds exactly the
same feature vector that build_dataset.py produced at training time, then scores
it. Sharing the feature code with the training script (rather than
reimplementing it here) is deliberate — any drift between training-time and
inference-time features would silently wreck accuracy, and that class of bug is
miserable to track down.

Threshold guidance: this is a RELEASE DETECTOR feeding a physics pipeline that
independently validates arc, height, and rim geometry. A false trigger is cheap
(downstream validation discards it); a missed release is unrecoverable (the shot
never enters the pipeline at all). So run it recall-favoring — lower threshold
than you'd pick for a standalone classifier.
"""

from collections import deque

import numpy as np
import joblib

from build_dataset import window_features, WINDOW_FRAMES


class LiveShotClassifier:
    def __init__(self, model_path, fps, threshold=0.20):
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.feature_names = bundle['feature_names']
        self.fps = fps
        self.threshold = threshold
        # Buffer needs to hold a full feature window.
        self.buffer = deque(maxlen=WINDOW_FRAMES + 4)
        self.last_proba = 0.0

    def update(self, keypoints):
        """Push this frame's (17,2) keypoints. Pass None if no person detected."""
        if keypoints is None:
            self.buffer.append(np.full((17, 2), np.nan, dtype=np.float32))
        else:
            self.buffer.append(np.asarray(keypoints, dtype=np.float32))

    def shot_probability(self):
        """Probability that a shot release is happening at the current frame."""
        if len(self.buffer) < WINDOW_FRAMES:
            return 0.0
        arr = np.stack(self.buffer)
        res = window_features(arr, len(arr) - 1, self.fps)
        if res is None:
            return 0.0
        feat, _ = res
        try:
            self.last_proba = float(self.model.predict_proba(feat.reshape(1, -1))[0, 1])
        except Exception:
            return 0.0
        return self.last_proba

    def is_release(self):
        return self.shot_probability() >= self.threshold

    def reset(self):
        self.buffer.clear()
        self.last_proba = 0.0
