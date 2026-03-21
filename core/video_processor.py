import cv2
import mediapipe as mp
from utils import calculate_angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_video_debug(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Convert from BGR to RGB so data is valid for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for a Squat (Hip, Knee, Ankle)
            hip = [max(0, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x), landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate Angle
            angle = calculate_angle(hip, knee, ankle)

            # Draw "Dummy" HUD
            h, w, _ = frame.shape
            cv2.putText(frame, f"Knee Angle: {int(angle)}", 
                        (int(knee[0]*w), int(knee[1]*h)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('RepRight HUD Development', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()