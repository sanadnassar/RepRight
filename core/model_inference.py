# Aref
import joblib
import numpy as np
import os

#load once when file is imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "..", "machine_learning", "squat_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "machine_learning", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
le    = joblib.load(ENCODER_PATH)

#takes 3 angles(knee, hip, back)
def predict_form(knee_angle, hip_angle, back_angle):
    
    # mirror the exact feature order from train_model.py
    features = np.array([[
        back_angle,   # right_elbow_right_shoulder_right_hip
        back_angle,   # left_elbow_left_shoulder_left_hip
        hip_angle,    # right_knee_mid_hip_left_knee
        knee_angle,   # right_hip_right_knee_right_ankle
        knee_angle    # left_hip_left_knee_left_ankle
    ]])

    prediction_encoded = model.predict(features)[0]
    probability        = model.predict_proba(features)[0]
    confidence         = float(np.max(probability))
    label              = le.inverse_transform([prediction_encoded])[0]

    #convert to 0-100 score
    if label == "good":
        score = int(50 + confidence * 50)   # 50-100
    else:
        score = int((1 - confidence) * 50)  # 0-50

    return label, score, confidence

#test
if __name__ == "__main__":
    label, score = predict_form(knee_angle=88, hip_angle=92, back_angle=60)
    print(f"Label: {label}, Score: {score}/100")