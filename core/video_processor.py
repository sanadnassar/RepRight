import cv2
import mediapipe as mp
import numpy as np
import tempfile
from collections import deque
from utils import calculate_angle, get_feedback, detect_heel_lift, detect_back_rounding
from core.model_inference import predict_form


#  MEDIAPIPE SETUP
#------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.2,
    model_complexity=1,
    smooth_landmarks=True
)

#feet skeleton added
CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32)
]
KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Score smoother prevents skeleton getting interrupted
#-----------------------------------------------------
class ScoreSmoother:
    def __init__(self, window=10):
        self.window = deque(maxlen=window)

    def update(self, score):
        self.window.append(score)
        return int(sum(self.window) / len(self.window))

#UI Helpers
#-----------
def get_score_colour(score, label=""):
    if label == "ready":  return (180, 180, 180)  # gray
    if label == "good":   return (0, 255, 100)     # green
    if label == "average": return (0, 200, 255)    # yellow
    if label == "bad":    return (0, 60, 255)      # red
    return (180, 180, 180)  # fallback gray

def get_pixel(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def draw_skeleton(frame, landmarks, score, label):
    h, w = frame.shape[:2]
    colour = get_score_colour(score, label)
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
    colour = get_score_colour(score, label)

    #left/main panel above video
    #-----------------------------
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



def detect_knee_cave(lm, k_ang):
    knee_width  = abs(lm[25].x - lm[26].x)
    ankle_width = abs(lm[27].x - lm[28].x)
    return knee_width < ankle_width * 0.85 and k_ang < 130



# Main Video processor
#-------------------------
def process_video(video_file, exercise, progress_callback):


    if exercise != "SQUATS":
        raise ValueError(f"{exercise} analysis coming soon! Currently only SQUATS are supported.")

    progress_callback(5, "Initializing analysis engine...")

    # write uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        input_path = tfile.name

    cap = cv2.VideoCapture(input_path)
    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

    output_path  = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    cap          = cv2.VideoCapture(input_path)
    fps          = int(cap.get(cv2.CAP_PROP_FPS))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    #Trackers 
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

    # Warning frequencyfor final report
    warning_counts = {
        "HEELS LIFTED": 0,
        "KNEES CAVING": 0,
        "BACK ROUNDED": 0,
        "LOW DEPTH":    0,
    }

    # Defaults so draw functions dont crash on first frame
    label           = "ready"
    score           = 50
    reasons         = []
    latest_feedback = ""
    smoother        = ScoreSmoother(window=10)
    
    frames_in_squat = 0
    SQUAT_WARMUP = 6  

    locked_label = "ready"
    locked_score = 50

    good_reps = 0
    confidence_scores = []


    last_known_score = 50
    last_known_label = "ready"

    consecutive_no_detection = 0
    MAX_NO_DETECTION_FRAMES = int(fps * 2)
    #frame Loop
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
            consecutive_no_detection = 0
            #landmark extraction
            l_sh    = [lm[11].x, lm[11].y]
            r_sh    = [lm[12].x, lm[12].y]
            l_hip   = [lm[23].x, lm[23].y]
            r_hip   = [lm[24].x, lm[24].y]
            l_knee  = [lm[25].x, lm[25].y]
            r_knee  = [lm[26].x, lm[26].y]
            l_ankle = [lm[27].x, lm[27].y]
            r_ankle = [lm[28].x, lm[28].y]

            #angle acalculation
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

            #biomechanical flags
            # using utils functions
            heel_lifted  = detect_heel_lift(lm)
            back_rounded = detect_back_rounding(lm)
            knee_caving  = detect_knee_cave(lm, k_ang) # Local — needs k_ang
            low_depth    = k_ang > 105                 # Not deep enough

            #rep state machine
            if k_ang < DEPTH_THRESHOLD and rep_state == "up":
                rep_state = "down"
                latest_feedback = get_feedback(k_ang, h_ang, b_ang, label)
                locked_label = label
                locked_score = score

            elif k_ang > DEPTH_THRESHOLD and rep_state == "down":
                rep_state = "up"
                rep_count += 1
                if current_rep_scores:
                    rep_avg = int(sum(current_rep_scores) / len(current_rep_scores))
                    rep_scores.append(rep_avg)
                    current_rep_scores = []
                # Count as good rep if minimum depth was reached this rep
                if knee_angles and min(knee_angles[-20:]) < 95: # check last 20 frames for depth
                    good_reps += 1

            # Active Rep guard
            #only score/track when person is actually squatting
            if k_ang < 145:
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
                frames_in_squat += 1

                if frames_in_squat <= SQUAT_WARMUP or k_ang > 110:
                    label = "ready"
                elif rep_state == "down":
                    if score >= 80:   label = "good"
                    elif score >= 60: label = "average"
                    else:             label = "bad"
                else:
                    # Ascending hold locked label
                    label = locked_label
                    score = locked_score

                # Update trackers
                if label != "ready":
                    total_person_frames += 1
                    all_scores.append(score)
                    confidence_scores.append(confidence)
                    last_known_score = score
                    last_known_label = label
                knee_angles.append(k_ang)
                back_angles.append(b_ang)
                current_rep_scores.append(score)

                if label == "good":
                    good_frames += 1

                # Count warnings for final report
                for reason in reasons:
                    if reason in warning_counts:
                        warning_counts[reason] += 1

            else:
                label = "ready"
                score = 50
                reasons = []
                frames_in_squat = 0
                locked_label = "ready"
                locked_score = 50
                smoother.window.clear()

            draw_skeleton(frame, lm, score, label)
            draw_hud(frame, k_ang, h_ang, b_ang, label, score, rep_count, reasons)
        else:
            consecutive_no_detection += 1
            if consecutive_no_detection > MAX_NO_DETECTION_FRAMES:
                cap.release()
                out.release()
                raise ValueError(
                    "Tracking was lost for more than 2 seconds. "
                    "Ensure your full body stays in frame throughout the video."
                )
            out.write(frame)
            continue

        

        out.write(frame)

    cap.release()
    out.release()

    # Final stats
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

    avg_confidence = int((sum(confidence_scores) / len(confidence_scores)) * 100) if confidence_scores else 0
    avg_knee_at_depth = int(min(knee_angles)) if knee_angles else 0

    if depth_pct < 75:
        depth_label = ""
    elif 75 <= depth_pct <= 100:
        depth_label = "Ideal"
   

    if avg_score >= 75:
        verdict = "Good"
    elif avg_score >= 55:
        verdict = "Decent"
    else:
        verdict = "Bad"

    return output_path, {
        "score":          avg_score,
        "verdict":        verdict,
        "total_reps":     rep_count,
        "good_reps":      good_reps,
        "depth_pct":      f"{depth_pct}%",
        "depth_angle":    f"{avg_knee_at_depth}°",
        "depth_label":    depth_label,
        "avg_confidence": f"{avg_confidence}%",
        "rep_scores":     rep_scores,
        "issue":          most_common_issue,
        "all_warnings":   warning_counts,
        "feedback":       latest_feedback,
    }