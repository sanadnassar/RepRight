import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle, get_feedback
from core.model_inference import predict_form
import tempfile

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
# skeleton bone pairs
CONNECTIONS = [
    (11, 12),   # shoulders
    (11, 23),   # left shoulder → left hip
    (12, 24),   # right shoulder → right hip
    (23, 24),   # hips
    (23, 25),   # left hip → left knee
    (25, 27),   # left knee → left ankle
    (24, 26),   # right hip → right knee
    (26, 28),   # right knee → right ankle
]

KEY_JOINTS = [11,12,23,24,25,26,27,28]

def get_score_colour(score):
    # returns rgb colour based on score
    if score >= 75:
        return (0,255,100)
    elif score >= 50:
        return (0, 200, 500)
    else:
        return (0, 60, 255)
    
def get_pixel(landmark, w, h):
    #convert 0-1 coords to pixel coords
    return (int(landmark.x * w), int(landmark.y * h))

def draw_skeleton(frame, landmarks, score):
    h, w = frame.shape [:2]
    colour = get_score_colour(score)

    #bones

    for start_idx, end_idx in CONNECTIONS:
        p1 = get_pixel(landmarks[start_idx], w, h)
        p2 = get_pixel(landmarks[end_idx],w,h)
        cv2.line(frame, p1, p2, colour, 3, cv2.LINE_AA)

    for idx in KEY_JOINTS:
        pt = get_pixel(landmarks[idx], w, h)
        cv2.circle(frame, pt, 6, colour,        -1)
        cv2.circle(frame, pt, 8, (255,255,255),  1)


def draw_hud(frame, knee_angle, hip_angle, back_angle,
             label, score, confidence, feedback, rep_count):
    h, w = frame.shape[:2]
    colour = get_score_colour(score)

    # semi-transparent dark panel top-left
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # HUD text
    lines = [
        f"RepRight",
        f"Reps: {rep_count}",
        f"Score: {score}/100",
        f"Confidence: {int(confidence*100)}%",
        f"Knee: {int(knee_angle)}  Hip: {int(hip_angle)}  Back: {int(back_angle)}",
        f"{label.upper()}",
    ]

    y = 38
    for i, line in enumerate(lines):
        font_scale = 0.8 if i != 0 else 1.1
        thickness  = 2   if i != 0 else 3
        col        = colour if i == 5 else (255, 255, 255)
        cv2.putText(frame, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, col, thickness, cv2.LINE_AA)
        y += 34

    # feedback text bottom of frame
    cv2.putText(frame, feedback, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, colour, 2, cv2.LINE_AA)


def process_video(video_path, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    # rep counting state
    rep_count  = 0
    rep_state  = "up"      # "up" or "down"
    DEPTH_THRESHOLD = 95   # degrees — below this = "down" position

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # extract coords — both sides
            l_hip    = [lm[23].x, lm[23].y]
            l_knee   = [lm[25].x, lm[25].y]
            l_ankle  = [lm[27].x, lm[27].y]
            r_hip    = [lm[24].x, lm[24].y]
            r_knee   = [lm[26].x, lm[26].y]
            r_ankle  = [lm[28].x, lm[28].y]
            l_shoulder = [lm[11].x, lm[11].y]
            r_shoulder = [lm[12].x, lm[12].y]

            # calculate angles — average both sides
            l_knee_angle = calculate_angle(l_hip,    l_knee, l_ankle)
            r_knee_angle = calculate_angle(r_hip,    r_knee, r_ankle)
            knee_angle   = (l_knee_angle + r_knee_angle) / 2

            l_hip_angle  = calculate_angle(l_shoulder, l_hip, l_knee)
            r_hip_angle  = calculate_angle(r_shoulder, r_hip, r_knee)
            hip_angle    = (l_hip_angle + r_hip_angle) / 2

            back_angle   = calculate_angle(
                [(l_shoulder[0]+r_shoulder[0])/2,
                 (l_shoulder[1]+r_shoulder[1])/2],
                [(l_hip[0]+r_hip[0])/2,
                 (l_hip[1]+r_hip[1])/2],
                l_knee
            )

            # rep counting state machine
            if knee_angle < DEPTH_THRESHOLD and rep_state == "up":
                rep_state = "down"
            elif knee_angle > DEPTH_THRESHOLD and rep_state == "down":
                rep_state = "up"
                rep_count += 1   # completed a full rep

            # ML prediction
            label, score, confidence = predict_form(
                knee_angle, hip_angle, back_angle
            )

            #feedback
            feedback = get_feedback(knee_angle, hip_angle, back_angle, label)

            #draw everything
            draw_skeleton(frame, lm, score)
            draw_hud(frame, knee_angle, hip_angle, back_angle,
                     label, score, confidence, feedback, rep_count)

        out.write(frame)

    cap.release()
    out.release()
    return output_path

