import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time
from utils import calculate_angle, get_feedback
from core.model_inference import predict_form

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28]

def get_score_colour(score):
    if score >= 75: return (0, 255, 100)
    elif score >= 50: return (0, 200, 255)
    else: return (0, 60, 255)

def get_pixel(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def draw_skeleton(frame, landmarks, score):
    h, w = frame.shape[:2]
    colour = get_score_colour(score)
    for start_idx, end_idx in CONNECTIONS:
        p1, p2 = get_pixel(landmarks[start_idx], w, h), get_pixel(landmarks[end_idx], w, h)
        cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)
    for idx in KEY_JOINTS:
        pt = get_pixel(landmarks[idx], w, h)
        cv2.circle(frame, pt, 6, colour, -1)
        cv2.circle(frame, pt, 8, (255, 255, 255), 1)

def draw_hud(frame, knee_angle, hip_angle, back_angle, label, score, confidence, feedback, rep_count):
    h, w = frame.shape[:2]
    colour = get_score_colour(score)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    lines = ["RepRight", f"Reps: {rep_count}", f"Score: {score}/100", 
             f"Conf: {int(confidence*100)}%", f"K:{int(knee_angle)} H:{int(hip_angle)} B:{int(back_angle)}", label.upper()]
    y = 38
    for i, line in enumerate(lines):
        font_scale, thickness = (1.1, 3) if i == 0 else (0.8, 2)
        col = colour if i == 5 else (255, 255, 255)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness, cv2.LINE_AA)
        y += 34
    cv2.putText(frame, feedback, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)

def process_video(video_file, exercise, progress_callback):
    # GUARD: Only SQUATS supported for now
    if exercise != "SQUATS":
        raise ValueError(f"{exercise} analysis coming soon! Currently only SQUATS are supported.")

    progress_callback(5, "Initializing analysis engine...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        input_path = tfile.name

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ISSUE 1 FIX: mp4v codec for cross-platform compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Real-time trackers
    rep_count, rep_state, DEPTH_THRESHOLD = 0, "up", 95
    current_frame, good_frames, total_person_frames = 0, 0, 0
    all_scores, knee_angles, back_angles = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_frame += 1

        if current_frame % 10 == 0:
            progress_callback(5 + int((current_frame / total_frames) * 90), f"Analyzing Frame {current_frame}/{total_frames}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            total_person_frames += 1
            lm = results.pose_landmarks.landmark

            l_hip, l_knee, l_ankle = [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y]
            r_hip, r_knee, r_ankle = [lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y]
            l_sh, r_sh = [lm[11].x, lm[11].y], [lm[12].x, lm[12].y]

            k_ang = (calculate_angle(l_hip, l_knee, l_ankle) + calculate_angle(r_hip, r_knee, r_ankle)) / 2
            h_ang = (calculate_angle(l_sh, l_hip, l_knee) + calculate_angle(r_sh, r_hip, r_knee)) / 2
            b_ang = calculate_angle([(l_sh[0]+r_sh[0])/2, (l_sh[1]+r_sh[1])/2], [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2], l_knee)

            knee_angles.append(k_ang)
            back_angles.append(b_ang)

            if k_ang < DEPTH_THRESHOLD and rep_state == "up": rep_state = "down"
            elif k_ang > DEPTH_THRESHOLD and rep_state == "down":
                rep_state = "up"
                rep_count += 1

            # ISSUE 3 CHECK: Returns 3 values
            label, score, confidence = predict_form(k_ang, h_ang, b_ang)
            all_scores.append(score)
            if label == "good": good_frames += 1
            
            draw_skeleton(frame, lm, score)
            draw_hud(frame, k_ang, h_ang, b_ang, label, score, confidence, get_feedback(k_ang, h_ang, b_ang, label), rep_count)

        out.write(frame)

    cap.release()
    out.release()

    # ISSUE 1 & 2 FIX: Calculating real stats
    avg_score = int(sum(all_scores) / len(all_scores)) if all_scores else 0
    good_pct = int((good_frames / total_person_frames) * 100) if total_person_frames > 0 else 0
    avg_knee = int(sum(knee_angles) / len(knee_angles)) if knee_angles else 0
    avg_back = int(sum(back_angles) / len(back_angles)) if back_angles else 0

    return output_path, {"score": avg_score, "reps": rep_count, "good_pct": f"{good_pct}%", "avg_knee": f"{avg_knee}°", "avg_back": f"{avg_back}°"}

