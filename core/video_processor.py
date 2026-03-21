import cv2
import mediapipe as mp
import numpy as np
import tempfile
from collections import deque
from utils import calculate_angle, get_feedback, detect_heel_lift, detect_back_rounding
from core.model_inference import predict_form

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2  # Higher precision landmark tracking
)

# Added feet connections (27-32) for full lower body skeleton
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32)
]
KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# ─────────────────────────────────────────────
#  SCORE SMOOTHER — Prevents flickering scores
# ─────────────────────────────────────────────
class ScoreSmoother:
    def __init__(self, window=10):
        self.window = deque(maxlen=window)

    def update(self, score):
        self.window.append(score)
        return int(sum(self.window) / len(self.window))

# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def get_score_colour(score):
    if score >= 75:   return (0, 255, 100)   # Green
    elif score >= 50: return (0, 200, 255)   # Yellow
    else:             return (0, 60, 255)    # Red

def get_pixel(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def draw_skeleton(frame, landmarks, score):
    h, w = frame.shape[:2]
    colour = get_score_colour(score)
    for start_idx, end_idx in CONNECTIONS:
        p1 = get_pixel(landmarks[start_idx], w, h)
        p2 = get_pixel(landmarks[end_idx], w, h)
        cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)
    for idx in KEY_JOINTS:
        pt = get_pixel(landmarks[idx], w, h)
        cv2.circle(frame, pt, 6, colour, -1)
        cv2.circle(frame, pt, 8, (255, 255, 255), 1)

def draw_hud(frame, knee_angle, hip_angle, back_angle,
             label, score, rep_count, warnings):
    h, w = frame.shape[:2]
    colour = get_score_colour(score)

    # --- Left Panel: Main Stats ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 195), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        ("RepRight",                                           (255, 255, 255), 1.1, 3),
        (f"Reps: {rep_count}",                                 (255, 255, 255), 0.8, 2),
        (f"K:{int(knee_angle)}  H:{int(hip_angle)}  B:{int(back_angle)}",
                                                               (200, 200, 200), 0.6, 1),
        (f"Score: {score}/100",                                colour,          0.85, 2),
        (label.upper(),                                        colour,          0.9,  2),
    ]
    y = 40
    for text, col, sc, th in lines:
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
        y += 32

    # --- Right Panel: Live Warnings (only shown during active squat) ---
    if warnings:
        panel_h = 35 + len(warnings) * 30
        warn_overlay = frame.copy()
        cv2.rectangle(warn_overlay, (w - 285, 10), (w - 10, 10 + panel_h), (0, 0, 40), -1)
        cv2.addWeighted(warn_overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "!!! WARNINGS !!!", (w - 275, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
        for i, warn in enumerate(warnings):
            cv2.putText(frame, f"- {warn}", (w - 275, 65 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  KNEE CAVE CHECK
#  (Can't be in utils.py since it needs 2 angles)
# ─────────────────────────────────────────────
def detect_knee_cave(lm, k_ang):
    """
    Knees should be at least as wide as ankles.
    If knees are 15% narrower than ankles during squat = caving.
    """
    knee_width  = abs(lm[25].x - lm[26].x)
    ankle_width = abs(lm[27].x - lm[28].x)
    return knee_width < ankle_width * 0.85 and k_ang < 130


# ─────────────────────────────────────────────
#  MAIN VIDEO PROCESSOR
# ─────────────────────────────────────────────
def process_video(video_file, exercise, progress_callback):

    if exercise != "SQUATS":
        raise ValueError(f"{exercise} analysis coming soon! Currently only SQUATS are supported.")

    progress_callback(5, "Initializing analysis engine...")

    # Write uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        input_path = tfile.name

    output_path  = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    cap          = cv2.VideoCapture(input_path)
    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ── Trackers ──
    DEPTH_THRESHOLD    = 95          # Knee angle that counts as "at depth"
    rep_count          = 0
    rep_state          = "up"
    current_frame      = 0
    good_frames        = 0
    total_person_frames = 0

    all_scores         = []
    knee_angles        = []
    back_angles        = []
    rep_scores         = []          # Average score per completed rep
    current_rep_scores = []          # Buffer for current rep

    # Warning frequency — for final report
    warning_counts = {
        "HEELS LIFTED": 0,
        "KNEES CAVING": 0,
        "BACK ROUNDED": 0,
        "LOW DEPTH":    0,
    }

    # Defaults — so draw functions don't crash on first frame
    label           = "ready"
    score           = 50
    reasons         = []
    latest_feedback = ""
    smoother        = ScoreSmoother(window=10)

    # ── Frame Loop ──
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        if current_frame % 10 == 0:
            pct = 5 + int((current_frame / total_frames) * 90)
            progress_callback(pct, f"Analyzing Frame {current_frame}/{total_frames}")

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # ── 1. LANDMARK EXTRACTION ──
            l_sh    = [lm[11].x, lm[11].y]
            r_sh    = [lm[12].x, lm[12].y]
            l_hip   = [lm[23].x, lm[23].y]
            r_hip   = [lm[24].x, lm[24].y]
            l_knee  = [lm[25].x, lm[25].y]
            r_knee  = [lm[26].x, lm[26].y]
            l_ankle = [lm[27].x, lm[27].y]
            r_ankle = [lm[28].x, lm[28].y]

            # ── 2. ANGLE CALCULATIONS ──
            l_k_ang = calculate_angle(l_hip, l_knee, l_ankle)
            r_k_ang = calculate_angle(r_hip, r_knee, r_ankle)
            k_ang   = (l_k_ang + r_k_ang) / 2

            h_ang   = (
                calculate_angle(l_sh, l_hip, l_knee) +
                calculate_angle(r_sh, r_hip, r_knee)
            ) / 2

            b_ang   = calculate_angle(
                [(l_sh[0] + r_sh[0]) / 2,  (l_sh[1] + r_sh[1]) / 2],
                [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2],
                l_knee
            )

            # ── 3. BIOMECHANICAL FLAGS ──
            # Using Aref's utility functions where possible, inline where not
            heel_lifted  = detect_heel_lift(lm)        # From utils.py
            back_rounded = detect_back_rounding(lm)    # From utils.py
            knee_caving  = detect_knee_cave(lm, k_ang) # Local — needs k_ang
            low_depth    = k_ang > 105                 # Not deep enough

            # ── 4. REP STATE MACHINE ──
            if k_ang < DEPTH_THRESHOLD and rep_state == "up":
                rep_state = "down"
                latest_feedback = get_feedback(k_ang, h_ang, b_ang, label)

            elif k_ang > DEPTH_THRESHOLD and rep_state == "down":
                rep_state = "up"
                rep_count += 1

                # Save this rep's average score
                if current_rep_scores:
                    rep_avg = int(sum(current_rep_scores) / len(current_rep_scores))
                    rep_scores.append(rep_avg)
                    current_rep_scores = []

            # ── 5. ACTIVE REP GUARD ──
            # Only score/track when person is actually squatting
            if k_ang < 165:
                label, raw_score, confidence, reasons = predict_form(
                    knee_angle   = k_ang,
                    hip_angle    = h_ang,
                    back_angle   = b_ang,
                    heel_lifted  = heel_lifted,
                    back_rounded = back_rounded,
                    knee_caving  = knee_caving,
                    lack_of_depth = low_depth
                )

                # Smooth score to prevent flickering
                score = smoother.update(raw_score)

                # Update trackers
                total_person_frames += 1
                all_scores.append(score)
                knee_angles.append(k_ang)
                back_angles.append(b_ang)
                current_rep_scores.append(score)
                latest_feedback = get_feedback(k_ang, h_ang, b_ang, label)

                if label == "good":
                    good_frames += 1

                # Count warnings for final report
                for reason in reasons:
                    if reason in warning_counts:
                        warning_counts[reason] += 1

            else:
                # Clear warnings when standing — don't show stale alerts
                reasons = []

            # ── 6. DRAW ──
            draw_skeleton(frame, lm, score)
            draw_hud(frame, k_ang, h_ang, b_ang, label, score, rep_count, reasons)

        out.write(frame)

    cap.release()
    out.release()

    # ── FINAL STATS ──
    avg_score   = int(sum(all_scores) / len(all_scores)) if all_scores else 0
    good_pct    = int((good_frames / total_person_frames) * 100) if total_person_frames > 0 else 0

    # From Aref — Depth % and Consistency
    depth_frames = sum(1 for k in knee_angles if k < DEPTH_THRESHOLD)
    depth_pct    = int((depth_frames / len(knee_angles)) * 100) if knee_angles else 0
    consistency  = int(np.std(knee_angles)) if len(knee_angles) > 1 else 0

    # Most common issue across all frames
    most_common_issue = max(warning_counts, key=warning_counts.get)
    if warning_counts[most_common_issue] == 0:
        most_common_issue = "None — Great Form!"

    return output_path, {
        "score":        avg_score,
        "reps":         rep_count,
        "good_pct":     f"{good_pct}%",
        "depth":        f"{depth_pct}%",          # From Aref ✅
        "consistency":  f"{consistency}° std",    # From Aref ✅
        "rep_scores":   rep_scores,               # Per-rep breakdown ✅
        "issue":        most_common_issue,        # Biggest problem ✅
        "all_warnings": warning_counts,           # Full warning breakdown ✅
        "feedback":     latest_feedback,
    }