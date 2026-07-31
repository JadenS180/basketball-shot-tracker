"""
Static false-positive filter.

THE PROBLEM THIS SOLVES
Lowering the detection confidence threshold (needed to see the fast-moving ball)
also surfaces stationary background objects the model misreads as basketballs —
a rock, a shadow, a leaf. In this footage one such object at roughly (481,464)
was detected for hundreds of consecutive frames and repeatedly hijacked tracking.

THE INSIGHT
A basketball is essentially never stationary. Background junk essentially always
is. So: remember where detections keep appearing, and blacklist any location that
produces detections across many frames spanning a long time window.

This is much more reliable than trying to separate them by confidence, because a
static false positive can easily be detected with HIGHER confidence than a
genuinely blurry ball mid-flight.
"""

from collections import deque


class StaticFilter:
    def __init__(self, fps, radius_px=25, min_hits=25, window_s=6.0):
        """
        radius_px : detections within this distance count as "the same place"
        min_hits  : how many hits at one place before it's considered static
        window_s  : hits must span at least this long — a ball briefly hovering
                    near the rim shouldn't get blacklisted, but a rock sitting
                    there for six seconds should
        """
        self.radius = radius_px
        self.min_hits = min_hits
        self.window = window_s * fps
        self.history = deque(maxlen=4000)   # (frame, x, y)
        self.blacklist = []                 # (x, y)

    def update(self, frame_idx, x, y):
        self.history.append((frame_idx, x, y))

    def _is_blacklisted(self, x, y):
        for bx, by in self.blacklist:
            if abs(x - bx) < self.radius and abs(y - by) < self.radius:
                return True
        return False

    def refresh(self, frame_idx):
        """Recompute the blacklist from recent history. Call periodically."""
        recent = [h for h in self.history if frame_idx - h[0] <= self.window]
        clusters = []   # [x, y, count, first_frame, last_frame]
        for f, x, y in recent:
            placed = False
            for c in clusters:
                if abs(x - c[0]) < self.radius and abs(y - c[1]) < self.radius:
                    c[2] += 1
                    c[4] = max(c[4], f)
                    placed = True
                    break
            if not placed:
                clusters.append([x, y, 1, f, f])

        new_blacklist = []
        for cx, cy, count, first_f, last_f in clusters:
            # Static = many hits at one spot AND spread over a long stretch of time.
            # Requiring the time spread avoids blacklisting a ball that legitimately
            # sat near the rim for a few frames.
            if count >= self.min_hits and (last_f - first_f) >= self.window * 0.5:
                new_blacklist.append((cx, cy))
        self.blacklist = new_blacklist

    def accept(self, x, y):
        """True if this detection looks like a real (moving) ball."""
        return not self._is_blacklisted(x, y)

    def describe(self):
        if not self.blacklist:
            return "no static false positives detected"
        pts = ", ".join(f"({x:.0f},{y:.0f})" for x, y in self.blacklist)
        return f"{len(self.blacklist)} static object(s) being ignored: {pts}"
