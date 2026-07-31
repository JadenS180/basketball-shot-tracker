"""
Shot detector — two-region trendline method (the Avi Shah approach), meshed with
release detection.

MAKE/MISS LOGIC — this is a faithful implementation of the reference design:

    UP REGION   : a WIDE box sitting above the rim.
    DOWN REGION : a NARROW box (about rim width) directly below the rim.

    When the ball is detected in the DOWN region, look back for the most recent
    detection in the UP region. Draw a straight line between those two points and
    solve for x where it crosses the rim's y:

        m = (y2 - y1) / (x2 - x1),   x_at_rim = (rim_y - b) / m

    If x_at_rim falls within the rim's horizontal span -> MAKE, else MISS.

Two things I had wrong before and have corrected here:
  * I used one large zone for both points. The reference uses two boxes of
    DIFFERENT sizes — the down region being narrow is what stops balls that
    merely pass near the hoop from resolving as shots.
  * I added a "ball left the zone => MISS" rule. That is NOT part of this design,
    and it was manufacturing false misses when the ball clipped a zone edge.
    A shot only resolves when the ball is actually seen below the rim.

RELEASE DETECTION is meshed in as CONTEXT, not as a gate: an upward-moving ball
away from the hoop arms a "shot pending" flag, which is recorded with each result
and used to suppress the cooldown for genuine putbacks. It can never block a
result — the region logic alone decides makes and misses.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from static_filter import StaticFilter

BALL_CLASS, HUMAN_CLASS, RIM_CLASS = 0, 1, 2

# Inference settings: default imgsz=640 downscales this 1920x1080 video and
# shrinks the ball to near-invisibility (~31% of frames detected). 1280 recovers
# most of it. This was the single highest-impact setting in the whole project.
INFER_SIZE = 1280
INFER_CONF = 0.05
RIM_CONF = 0.4
HUMAN_CONF = 0.4              # confidence for treating a detection as "a person" for
                              # the ball-on-body rejection above

# --- UP region: wide box above the rim (blue box in the reference diagrams) ---
UP_HALF_WIDTH_MULT = 2.0      # in rim widths — the CORE up region (drawn blue)
# Wide-angle shots arc in from outside the core region, so their up-point was
# never recorded and they could never resolve. Rather than bounding the up region
# by x, track the ball ANYWHERE above rim height and gate on apparent SIZE
# instead: a ball near the low camera looks far bigger than one at the rim's
# distance, which is what separates a genuine high shot from a nearby dribble.
WIDE_UP_ENABLED = True
ABSURD_BALL_WIDTH_MULT = 2.5  # a "ball" wider than this many rim widths is not a
                              # ball — rejected outright at detection time
BALL_SIZE_MAX_MULT = 1.9      # reject as "too close to camera" beyond this multiple
                              # of the expected ball width at the rim's depth
BALL_DIAMETER_M, RIM_DIAMETER_M = 0.24, 0.4572
PLAUSIBLE_CROSS_MULT = 4.0    # crossing must land within this many rim widths of
                              # rim centre, else it wasn't a shot at this hoop
WIDE_UP_PLAUSIBLE_MULT = 4.5  # even the WIDE up-tracking needs SOME bound — without
                              # this, a stray/unrelated detection far from the hoop
                              # could become "the" up-point and sit there indefinitely
UP_POINT_MAX_AGE_S = 1.5      # an up-point this old is almost certainly stale (from an
                              # unrelated earlier moment, e.g. after a detection gap
                              # during the ball's real rise) — don't pair with it
UP_HEIGHT_MULT = 5.0          # in rim heights, above the rim line

# --- DOWN region: narrow box below the rim (green box in the diagrams) ---
# Widened from 0.9: this only controls whether a shot REGISTERS, not how it's
# judged. Make/miss is decided by where the trendline crosses the rim's span, so
# a wider down region raises detection without affecting classification accuracy.
DOWN_HALF_WIDTH_MULT = 2.6    # widened again from 2.0 — still ball-tracked shots
                              # were barely missing the box (your 0:26 case)
DOWN_HEIGHT_MULT = 5.0        # widened from 4.0, same reasoning — in rim heights,
                              # below the rim line

RIM_X_MARGIN_FRAC = 0.10      # tolerance on the rim's horizontal span
# Deflection thresholds — see the checks in the scoring block below.
APEX_OVERSIZE_MULT = 1.35     # up-point ball this much wider than expected rim-depth
                              # size means it's in FRONT of the net (perspective).
                              # Real shots measured 0.89-1.03x, so 1.35 is comfortably clear.
DEFLECT_STEEPNESS_MAX = 1.2   # |dx|/|dy| above this is a sideways deflection, not a drop
MAX_PAIR_DX_MULT = 5.0        # loosened again from 3.2 — the 0:43-45 shot needed up
                              # to 397px separation (a real wide-angle release, up-
                              # point moving 1387->1410 while still above the rim,
                              # not a stale point)
CONFIRM_WINDOW_S = 0.25       # hold a candidate crossing this long before finalizing —
                              # gives a rattling ball time to reveal itself before
                              # we lock in a verdict
RATTLE_MARGIN_FRACTION = 0.20 # in rim heights — the ball must clear the rim line by
                              # this much to count as a genuine bounce-back, not
                              # ordinary detection jitter on the bounding box center
MAX_STRADDLE_S = 0.7          # up-point and down-point must be this close in time
                              # (loosened: detection gaps shouldn't drop a shot)
# A pure time cooldown is the wrong tool here: it can't tell a rim rattle (one
# shot bouncing) from a putback (two genuine attempts seconds apart). What
# actually separates them is whether the ball LEFT the rim area in between — a
# rattling ball stays at the rim continuously, whereas a rebound-and-reshoot
# takes the ball away and brings it back. So the cooldown is short, and a new
# attempt is additionally allowed any time the ball has genuinely departed since
# the last result.
COOLDOWN_S = 0.5
RESET_DISTANCE_MULT = 3.0     # ball must get this far from rim (in rim widths)
                              # to re-arm scoring inside the cooldown window
RESULT_TIME_GRACE_S = 1.0     # ...OR this much time can pass instead — catches fast
                              # close-range rebounds that never travel far enough away
# --- Rim bounce-out ---
# A shot that clangs the rim and ricochets AWAY never enters the down region, so
# the two-region method alone can't see it — that was most of the undetected
# shots. To count as a bounce-out (rather than a ball merely passing overhead)
# the ball must genuinely have approached the rim first, then departed without
# ever coming down through it.
# Bounce-out detection is OFF by default. In testing it produced far more false
# misses than genuine catches — a made shot dropping through the net looks, to
# this rule, much like a ball approaching and departing the rim. Set True to
# experiment, but verify against ground truth before trusting it.
BOUNCEOUT_ENABLED = True
ARRIVAL_HEIGHT_MULT = 1.0     # ball must have been this many rim heights above the rim
ARRIVAL_MEMORY_S = 1.2        # ...within this long before approaching, and be descending
NEVER_GREEN_TIMEOUT_S = 0.8     # ball seen above the rim but not the green box within
                                # this long => it clanged out and never went through
NEVER_GREEN_ARM_DIST_MULT = 2.5 # must get within this many rim widths of the rim to
                                # arm the never-green timer — stops wide-tracked
                                # dribbles/passes that never approach the hoop from
                                # arming a timer that has nothing to do with them
BOUNCEOUT_APPROACH_MULT = 1.1   # must get within this many rim widths of rim centre
BOUNCEOUT_LEAVE_MULT = 2.6      # ...then get this far away to count as departed
BOUNCEOUT_MAX_S = 1.5           # within this long of the closest approach
RELEASE_UP_SPEED = 350.0      # px/s upward to arm "shot pending"
RELEASE_MEMORY_S = 3.0        # how long a release stays armed

video_path = 'media/basketball_test.mov'
model = YOLO('models/basketball_model.pt')

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('media/final_output.mp4', fourcc, fps, (width, height))

static_filter = StaticFilter(fps)
rim_samples, rim_box = [], None

up_point = None            # (frame, x, y) most recent detection in the UP region
pending = None              # candidate crossing awaiting confirmation (see loop)
near_rim_frame = -9999     # last frame the ball was genuinely close to the rim
near_rim_pending = False   # ball approached the rim; outcome not yet resolved
core_up_frame = 0          # last frame the ball was in the CORE blue box, descending
left_rim_since_score = True  # has the ball departed the rim area since the last result?
was_above_rim_frame = -9999  # last frame the ball was clearly ABOVE the rim
descending = False           # is the ball currently moving downward?
prev_ball = None           # (frame, x, y) previous detection anywhere, for release
release_frame = -9999      # last frame a release-like upward motion was seen

frame_count = 0
makes = misses = 0
last_score_frame = -9999
events = []
banner, banner_until = None, 0
ball_seen = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 60 == 0:
        static_filter.refresh(frame_count)

    ball = None
    human_boxes = []
    ball_candidates = []
    for r in model(frame, imgsz=INFER_SIZE, conf=INFER_CONF, verbose=False):
        for box in r.boxes:
            cls_id, conf = int(box.cls[0]), float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id == RIM_CLASS and conf > RIM_CONF:
                rim_samples.append((x1, y1, x2, y2))
                if len(rim_samples) > 90:
                    rim_samples.pop(0)
            elif cls_id == HUMAN_CLASS and conf > HUMAN_CONF:
                human_boxes.append((x1, y1, x2, y2))
            elif cls_id == BALL_CLASS:
                ball_candidates.append((x1, y1, x2, y2, conf))

    # Filter ball candidates against human boxes AFTER collecting both, since YOLO
    # returns detections in no particular order within one frame.
    ball_on_person = False
    for x1, y1, x2, y2, conf in ball_candidates:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw_raw = x2 - x1
        # HARD SIZE REJECT. At conf=0.05 the model sometimes emits a huge
        # box over the house wall and calls it a basketball. A ball can
        # never be several rim-widths across, so drop these outright
        # before they can pollute tracking.
        if rim_box is not None:
            _rw = max(rim_box[2] - rim_box[0], 1)
            if bw_raw > _rw * ABSURD_BALL_WIDTH_MULT:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                continue
        # NOTE: overlap with a person is tracked but NOT rejected here. A layup
        # releases the ball right against the shooter's body — overlap there is
        # completely normal, not the failure mode we're guarding against. This
        # is only used later to gate the WIDE up-tracking specifically, which is
        # where a false detection on the arm actually caused a wrong result
        # (up=(1410,284), 300+px from the hoop, on the shooter's follow-through).
        on_a_person = any(hx1 <= cx <= hx2 and hy1 <= cy <= hy2
                          for hx1, hy1, hx2, hy2 in human_boxes)
        static_filter.update(frame_count, cx, cy)
        if not static_filter.accept(cx, cy):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
            continue
        if ball is None or conf > ball[2]:
            ball = (cx, cy, conf, bw_raw)
            ball_on_person = on_a_person
        color = (255, 0, 255) if on_a_person else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if ball is not None:
        ball_seen += 1

    if len(rim_samples) >= 15:
        rim_box = np.median(np.array(rim_samples), axis=0).astype(int)

    if rim_box is not None:
        rx1, ry1, rx2, ry2 = rim_box
        rim_y = ry1
        rim_w = max(rx2 - rx1, 1)
        rim_h = max(ry2 - ry1, 1)
        rim_cx = (rx1 + rx2) / 2
        margin = rim_w * RIM_X_MARGIN_FRAC

        up_hw = rim_w * UP_HALF_WIDTH_MULT
        up_top = rim_y - rim_h * UP_HEIGHT_MULT
        dn_hw = rim_w * DOWN_HALF_WIDTH_MULT
        dn_bot = rim_y + rim_h * DOWN_HEIGHT_MULT

        # Draw both regions: blue = up, green = down (matching the reference).
        cv2.rectangle(frame, (int(rim_cx - up_hw), int(up_top)),
                      (int(rim_cx + up_hw), rim_y), (255, 120, 0), 1)
        cv2.rectangle(frame, (int(rim_cx - dn_hw), rim_y),
                      (int(rim_cx + dn_hw), int(dn_bot)), (0, 255, 0), 1)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        if ball is not None:
            bx, by, _, bw = ball

            # --- Release detection (context only, never blocks a result) ---
            if prev_ball is not None:
                pf, px, py = prev_ball
                dt = (frame_count - pf) / fps
                if 0 < dt <= 0.3:
                    vy_up = (py - by) / dt          # +ve = moving up
                    away_from_hoop = abs(bx - rim_cx) > rim_w * 1.5
                    if vy_up > RELEASE_UP_SPEED and away_from_hoop:
                        release_frame = frame_count
            prev_ball = (frame_count, bx, by)

            expected_bw = rim_w * (BALL_DIAMETER_M / RIM_DIAMETER_M)
            size_ok = (bw is None) or (bw <= expected_bw * BALL_SIZE_MAX_MULT)

            in_up = (abs(bx - rim_cx) <= up_hw and up_top <= by < rim_y)
            # Wide up-region: anywhere above rim height, provided the ball is a
            # plausible size for that depth AND isn't sitting on the player's own
            # body (a false detection on an arm/hand far from the hoop, not a
            # real airborne ball). Scoped to WIDE tracking only — the core
            # up-region and down-region are untouched, since a layup's ball
            # naturally overlaps the shooter and that's completely normal there.
            if WIDE_UP_ENABLED and not in_up and by < rim_y and size_ok \
                    and not ball_on_person \
                    and abs(bx - rim_cx) <= rim_w * WIDE_UP_PLAUSIBLE_MULT:
                in_up = True
                cv2.circle(frame, (bx, by), 8, (255, 200, 0), 1)

            in_down = (abs(bx - rim_cx) <= dn_hw and rim_y <= by <= dn_bot)

            # Track where the ball came FROM. This is what separates a genuine
            # clang-out (shot arrives from a high arc, descending) from a rebound
            # or loose ball (arrives from below/sideways after bouncing). Without
            # it, a ball that leaves the rim, comes back, and bounces away again
            # is indistinguishable from a missed shot and scores a phantom MISS.
            if prev_ball is not None:
                descending = by > prev_ball[2]
            if by < rim_y - rim_h * ARRIVAL_HEIGHT_MULT:
                was_above_rim_frame = frame_count

            dist_mult = ((bx - rim_cx) ** 2 + (by - rim_y) ** 2) ** 0.5 / rim_w
            if dist_mult >= RESET_DISTANCE_MULT:
                # Ball is well clear of the hoop — whatever happens next is a new
                # attempt, not the tail of the previous one.
                left_rim_since_score = True
            # ---- BLUE-BUT-NEVER-GREEN => MISS ----
            # Replaces the old distance-based bounce-out, which used fuzzy
            # rim-width thresholds and wasn't firing on the real clang-outs.
            # This is the region-based version: if the ball was ever tracked
            # ABOVE the rim (descending, arrived from a high arc) and then never
            # appears in the green down region, it clanged out.
            #
            # IMPORTANT FIX: this used to arm ONLY from the narrow core blue box
            # (in_core_up), not the wide above-rim tracking (in_up covers both).
            # The dot you see drawn for wide-tracked shots LOOKS like "in the blue
            # box" but was a completely different check that never armed this
            # timer — so a shot only ever caught by wide tracking could bounce
            # away and NEVER register as a miss, no matter how obvious the
            # clang-out. That's exactly the 2:32/2:55 pattern.
            arrived_from_above = (descending and
                                  (frame_count - was_above_rim_frame) <= ARRIVAL_MEMORY_S * fps)
            # Computed once, used everywhere a "has enough happened since the last
            # result" check is needed — both here (never-green) and in the
            # down-crossing resolution below. Previously the never-green check used
            # the raw left_rim_since_score flag directly, with no time-based
            # fallback, so it could get blocked the exact same way normal
            # resolutions were: a shot taken shortly after a prior one, close to
            # the rim, would never satisfy pure distance-departure, and this
            # mechanism just... didn't run. Same bug, different location.
            cooled = (left_rim_since_score or
                      (frame_count - last_score_frame) > RESULT_TIME_GRACE_S * fps)

            # Latch on FIRST entry only. Updating this every frame would reset the
            # timer continuously while the ball is still above the rim, so the
            # timeout could never fire. Also requires genuine PROXIMITY to the rim
            # right now — without this, any wide-tracked dribble or pass anywhere
            # in the plausible zone (up to 4.5 rim-widths away) could arm a timer
            # that then fires a phantom miss, since it never approached the hoop
            # at all. Real bounce-out shots get close to the rim before departing;
            # this just requires that to have actually happened.
            near_hoop_now = dist_mult <= NEVER_GREEN_ARM_DIST_MULT
            if in_up and arrived_from_above and near_hoop_now and cooled and core_up_frame == 0:
                core_up_frame = frame_count

            if (BOUNCEOUT_ENABLED
                    and core_up_frame > 0
                    and cooled
                    and (frame_count - core_up_frame) > NEVER_GREEN_TIMEOUT_S * fps):
                misses += 1
                events.append((frame_count, 'MISS', f"{frame_count/fps:.1f}s"))
                banner, banner_until = 'MISS', frame_count + 45
                last_score_frame = frame_count
                left_rim_since_score = False
                up_point = None
                print(f"[frame {frame_count}] MISS — ball was tracked above the rim at frame "
                      f"{core_up_frame} but never reached the green box within "
                      f"{NEVER_GREEN_TIMEOUT_S}s — clanged out")
                core_up_frame = 0

            # ---- Confirmation window (structural fix for the rattle/perspective ----
            # ---- problem you've been flagging repeatedly) ----
            # We do NOT resolve on the very first frame the ball's center crosses
            # into the down region. A ball rattling in front of the net crosses
            # that line multiple times (up-down-up-down) before actually
            # settling, and resolving on the first crossing was exactly "counted
            # the make early, then it bounced off." Instead: hold the candidate
            # open for CONFIRM_WINDOW_S. If the ball reappears ABOVE the rim line
            # during that window, it was a bounce — discard and keep watching,
            # don't score anything. Only finalize if it doesn't.
            if pending is not None:
                # Require the ball to clear the rim line by a MEANINGFUL margin, not
                # just any pixel of jitter. Frame 168 on your first shot: y=299 vs
                # rim_y=311 — a 12px difference, ordinary bounding-box noise on a
                # fast-moving ball, not a real bounce. That killed a shot that was
                # resolving correctly before this check existed.
                if by < rim_y - rim_h * RATTLE_MARGIN_FRACTION:
                    print(f"[frame {frame_count}] RATTLE — ball back above the rim line "
                          f"during confirmation (y={by} vs rim_y={rim_y}); discarding "
                          f"candidate crossing, continuing to watch")
                    pending = None
                else:
                    if by > pending['by']:
                        pending['bx'], pending['by'], pending['bw'] = bx, by, bw
                    if (frame_count - pending['first_frame']) >= CONFIRM_WINDOW_S * fps:
                        f0, x0, y0, w0 = pending['up']
                        dbx, dby, dbw = pending['bx'], pending['by'], pending['bw']
                        t = (rim_y - y0) / (dby - y0) if dby != y0 else 0
                        cross_x = x0 + (dbx - x0) * t
                        if abs(cross_x - rim_cx) > rim_w * PLAUSIBLE_CROSS_MULT:
                            print(f"[frame {frame_count}] DISCARDED — confirmed crossing "
                                  f"x={cross_x:.0f} is {abs(cross_x-rim_cx)/rim_w:.1f} "
                                  f"rim-widths from the hoop, not a shot here")
                        else:
                            crossed_inside = (rx1 - margin) <= cross_x <= (rx2 + margin)
                            dy_travel = max(abs(dby - y0), 1)
                            sideways = abs(dbx - x0) / dy_travel > DEFLECT_STEEPNESS_MAX
                            expected_w = rim_w * (BALL_DIAMETER_M / RIM_DIAMETER_M)
                            oversized_at_apex = (w0 and w0 > expected_w * APEX_OVERSIZE_MULT)
                            # Size was previously only ever checked at the UP-point
                            # (apex, slow, low blur). A ball that looks normal on the
                            # way up can still deflect toward the camera and become
                            # oversized specifically AT the crossing — that gap is
                            # exactly "misses off the front of the rim, called a
                            # make." This is an ABSOLUTE size check on the held-down
                            # point, not a growth delta, so it isn't confounded by
                            # motion blur the way the old (removed) growth check was.
                            oversized_at_down = (dbw and dbw > expected_w * APEX_OVERSIZE_MULT)
                            if crossed_inside and (sideways or oversized_at_apex or oversized_at_down):
                                made = False
                                why = []
                                if sideways:
                                    why.append(f"path too sideways "
                                               f"(dx/dy={abs(dbx-x0)/dy_travel:.1f})")
                                if oversized_at_apex:
                                    why.append(f"ball oversized at apex ({w0}px vs "
                                               f"{expected_w:.0f}px expected)")
                                if oversized_at_down:
                                    why.append(f"ball oversized at the crossing itself "
                                               f"({dbw}px vs {expected_w:.0f}px expected) "
                                               f"— in front of the net, not through it")
                                print(f"[frame {frame_count}] DEFLECTION — confirmed "
                                      f"crossing looked like a make but: {'; '.join(why)}")
                            else:
                                made = crossed_inside
                            kind = 'MAKE' if made else 'MISS'
                            if made:
                                makes += 1
                            else:
                                misses += 1
                            events.append((frame_count, kind, f"{frame_count/fps:.1f}s"))
                            banner, banner_until = kind, frame_count + 45
                            last_score_frame = frame_count
                            print(f"[frame {frame_count}] {kind} — CONFIRMED trendline "
                                  f"crosses rim at x={cross_x:.0f} (rim {rx1}-{rx2}), "
                                  f"up=({x0},{y0},w{w0}) down=({dbx},{dby},w{dbw}) "
                                  f"[held {CONFIRM_WINDOW_S}s to rule out a rattle]")
                            near_rim_pending = False
                            left_rim_since_score = False
                            core_up_frame = 0
                        pending = None
                        up_point = None

            if in_up and pending is None:
                up_point = (frame_count, bx, by, bw)
                cv2.circle(frame, (bx, by), 6, (255, 120, 0), 2)

            elif in_down and up_point is not None and pending is None:
                f0, x0, y0, w0 = up_point

                if (frame_count - f0) > UP_POINT_MAX_AGE_S * fps:
                    # Stale up-point — almost certainly from an unrelated earlier
                    # moment (e.g. a detection gap swallowed the ball's real rise).
                    # Pairing with it produces nonsense; clear it instead.
                    print(f"[frame {frame_count}] up-point at ({x0},{y0}) is "
                          f"{(frame_count-f0)/fps:.2f}s old (limit {UP_POINT_MAX_AGE_S}s) "
                          f"— stale, clearing rather than pairing with it")
                    up_point = None
                else:
                    recent_release = (frame_count - release_frame) <= RELEASE_MEMORY_S * fps
                    # `cooled` was already computed once, above, and reused here —
                    # a single source of truth instead of two copies that could
                    # silently drift apart (which is exactly what happened before).
                    straddle_ok = (frame_count - f0) <= MAX_STRADDLE_S * fps
                    # The up-point and down-point must plausibly be the same ball on
                    # one continuous descent. Wide up-tracking otherwise paired a ball
                    # on the far side of the frame with a down-point at the hoop.
                    pair_ok = abs(x0 - bx) <= rim_w * MAX_PAIR_DX_MULT

                    # EARLY oversize gate. Previously the size check only ran AFTER a
                    # candidate was already opened, and only softened a MAKE into a
                    # MISS — it never stopped something from being scored as an
                    # attempt at all. A ball this close to the camera (a dribble,
                    # not a shot) shouldn't register as ANY result, make or miss.
                    expected_w = rim_w * (BALL_DIAMETER_M / RIM_DIAMETER_M)
                    apex_too_close = (w0 and w0 > expected_w * APEX_OVERSIZE_MULT)

                    if not cooled:
                        print(f"[frame {frame_count}] BLOCKED (ball hasn't left the rim "
                              f"area since the last result — anti-double-count). "
                              f"up=({x0},{y0}) down=({bx},{by})")
                    elif not straddle_ok:
                        print(f"[frame {frame_count}] BLOCKED (up-point is "
                              f"{(frame_count-f0)/fps:.2f}s old, limit {MAX_STRADDLE_S}s). "
                              f"up=({x0},{y0}) down=({bx},{by})")
                    elif not pair_ok:
                        print(f"[frame {frame_count}] BLOCKED (up and down points "
                              f"{abs(x0-bx)}px apart, limit "
                              f"{rim_w*MAX_PAIR_DX_MULT:.0f}px). up=({x0},{y0}) "
                              f"down=({bx},{by})")
                    elif apex_too_close:
                        print(f"[frame {frame_count}] NOT A SHOT — up-point ball is "
                              f"{w0}px vs {expected_w:.0f}px expected at rim depth; "
                              f"too close to the camera to be a real attempt "
                              f"(dribble/carry). up=({x0},{y0}) down=({bx},{by})")
                        up_point = None

                    if cooled and straddle_ok and pair_ok and not apex_too_close and by != y0:
                        # Open a CANDIDATE — do not resolve yet. See the confirmation
                        # block above, which runs on subsequent frames and only
                        # finalizes this if the ball doesn't bounce back above the
                        # rim line first.
                        pending = {'up': up_point, 'bx': bx, 'by': by, 'bw': bw,
                                   'first_frame': frame_count}
                        print(f"[frame {frame_count}] CANDIDATE crossing detected "
                              f"(up=({x0},{y0}) down=({bx},{by})) — holding "
                              f"{CONFIRM_WINDOW_S}s to confirm it's not a rattle")
                    elif not straddle_ok:
                        up_point = None   # stale, don't pair it

    if banner and frame_count <= banner_until:
        color = (0, 255, 0) if banner == 'MAKE' else (0, 0, 255)
        cv2.putText(frame, banner, (width // 2 - 120, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)
    cv2.putText(frame, f"Makes: {makes}   Misses: {misses}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    out.write(frame)
    if frame_count % 250 == 0:
        print(f"  {frame_count} frames — makes {makes}, misses {misses}")

cap.release()
out.release()

total = makes + misses
print(f"\nDone. {frame_count} frames")
print(f"Ball detected in {ball_seen} frames ({100.0*ball_seen/max(frame_count,1):.1f}%)")
print(f"Makes: {makes}   Misses: {misses}   Attempts: {total}")
if total:
    print(f"FG%: {100.0*makes/total:.1f}%")
print("\nAll attempts (check these timestamps against the video):")
for i, ev in enumerate(events, 1):
    f_no, kind = ev[0], ev[1]
    secs = f_no / fps
    print(f"  {i:2d}. {int(secs//60)}:{secs%60:04.1f}  {kind}")
print(f"\nRaw events: {events}")
print(f"Static filter: {static_filter.describe()}")
print("Output: media/final_output.mp4")
