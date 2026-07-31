"""
Reusable tracking primitives for the basketball shot tracker.

- KalmanBallTracker: constant-acceleration (gravity-aware) Kalman filter for the ball.
- TrajectoryBuffer: rolling window of recent (measured or predicted) ball states.
- HoopModel: smoothed hoop location + hoop-relative ("rim-width unit") coordinate
  conversions + the rim-cylinder make/miss test.
- fit_parabola: least-squares quadratic fit used to smooth/validate a trajectory segment.

Everything that used to be a hardcoded pixel constant in shot_detector.py is now
derived from the hoop's own measured size and the video's fps, so the same code
should behave reasonably across different camera distances/angles without
re-tuning magic numbers by hand.
"""

import statistics
from collections import deque

import cv2
import numpy as np


# ---- Physical constants (real world), used to calibrate pixel-space physics ----
GRAVITY_M_S2 = 9.8
RIM_DIAMETER_M = 0.4572   # 18in regulation rim
BALL_DIAMETER_M = 0.24    # standard basketball


class KalmanBallTracker:
    """
    4-state Kalman filter: [x, y, vx, vy], with gravity injected as a control
    input each predict() step (constant downward acceleration in image-y).

    Call predict() every frame regardless of whether a detection exists —
    that's what gives us "keep tracking through occlusion" for free. Call
    correct(x, y) only on frames with a real measurement.
    """

    def __init__(self, dt, gravity_px_per_frame2,
                 process_noise=0.8, measurement_noise=4.0):
        self.dt = dt
        self.gravity = gravity_px_per_frame2
        self.kf = cv2.KalmanFilter(4, 2, 1)  # state=4, measurement=2, control=1

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.controlMatrix = np.array([
            [0.0],
            [0.5 * dt * dt],
            [0.0],
            [dt],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.frames_since_measurement = 0

    def reset(self, x, y, vx=0.0, vy=0.0):
        """Re-seed the filter — call this at shot release/RISE detection."""
        self.kf.statePost = np.array([[x], [y], [vx], [vy]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = True
        self.frames_since_measurement = 0

    def predict(self):
        """Advance one frame. Always call this, even with no measurement."""
        control = np.array([[self.gravity]], dtype=np.float32)
        state = self.kf.predict(control)
        self.frames_since_measurement += 1
        return float(state[0, 0]), float(state[1, 0])

    def correct(self, x, y):
        """Fuse a real detection into the filter."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        self.frames_since_measurement = 0

    @property
    def velocity(self):
        vx, vy = float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])
        return vx, vy

    @property
    def position(self):
        x, y = float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])
        return x, y


class TrajectoryBuffer:
    """Rolling window of recent ball states, each tagged as measured or predicted."""

    def __init__(self, maxlen=30):
        self.buffer = deque(maxlen=maxlen)

    def add(self, frame_idx, t, x, y, vx, vy, measured):
        self.buffer.append({
            "frame": frame_idx, "t": t, "x": x, "y": y,
            "vx": vx, "vy": vy, "measured": measured,
        })

    def recent(self, n=None):
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def measured_fraction(self, n=None):
        """Confidence proxy: what fraction of recent points were real detections
        vs. Kalman-predicted? Useful for logging how much we're "trusting the model"
        on any given shot."""
        pts = self.recent(n)
        if not pts:
            return 0.0
        return sum(1 for p in pts if p["measured"]) / len(pts)

    def clear(self):
        self.buffer.clear()


def fit_parabola(points):
    """
    Least-squares fit y = a*x^2 + b*x + c to a list of {"x":..,"y":..} points.
    Returns (a, b, c) or None if underdetermined/degenerate.
    Used to validate "is this actually a ballistic arc" and to smooth noisy
    stretches of a trajectory rather than trusting a single noisy point.
    """
    if len(points) < 4:
        return None
    xs = np.array([p["x"] for p in points], dtype=np.float64)
    ys = np.array([p["y"] for p in points], dtype=np.float64)
    if np.ptp(xs) < 1e-3:  # degenerate: near-vertical, x barely varies
        return None
    try:
        a, b, c = np.polyfit(xs, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(a), float(b), float(c)


class HoopModel:
    """
    Smoothed hoop location (median over a rolling window of confident detections,
    so one noisy frame can't corrupt everything downstream) plus hoop-relative
    ("rim-width units") thresholds so nothing here is a hardcoded pixel constant —
    it all scales with how big the rim appears in THIS video.
    """

    def __init__(self, fps, smooth_window_s=1.5, warmup_s=0.5):
        self.fps = fps
        self.samples = deque(maxlen=int(fps * smooth_window_s))
        self.warmup_frames = int(fps * warmup_s)
        self.box = None          # (x1, y1, x2, y2), smoothed
        self.announced = False

    def add_sample(self, x1, y1, x2, y2):
        self.samples.append((x1, y1, x2, y2))
        if len(self.samples) >= self.warmup_frames:
            xs1 = sorted(s[0] for s in self.samples)
            ys1 = sorted(s[1] for s in self.samples)
            xs2 = sorted(s[2] for s in self.samples)
            ys2 = sorted(s[3] for s in self.samples)
            self.box = (
                int(statistics.median(xs1)), int(statistics.median(ys1)),
                int(statistics.median(xs2)), int(statistics.median(ys2)),
            )

    @property
    def ready(self):
        return self.box is not None

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def center_x(self):
        return (self.box[0] + self.box[2]) / 2

    @property
    def top(self):
        return self.box[1]

    @property
    def px_per_meter(self):
        return self.width / RIM_DIAMETER_M

    @property
    def gravity_px_per_s2(self):
        return GRAVITY_M_S2 * self.px_per_meter

    @property
    def expected_ball_width_px(self):
        return self.width * (BALL_DIAMETER_M / RIM_DIAMETER_M)

    # ---- Rim cylinder model ----
    # We don't have real depth, so the "cylinder" is approximated in image space:
    # a vertical column spanning [rim_top, net_bottom] in y, and a horizontal
    # radius around the rim's center in x. All distances scale off rim width.

    def cylinder_y_range(self, net_depth_fraction=0.9):
        net_bottom = self.box[3] + int(self.height * net_depth_fraction)
        return self.top, net_bottom

    def cylinder_radius(self, margin_fraction=0.20):
        return (self.width / 2) + self.width * margin_fraction

    def in_cylinder_x(self, x, margin_fraction=0.20):
        r = self.cylinder_radius(margin_fraction)
        return abs(x - self.center_x) <= r

    def plausible_shot_x(self, x, multiplier=5):
        return abs(x - self.center_x) <= self.width * multiplier
