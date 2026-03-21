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
def predict_form(knee_angle, hip_angle, back_angle, heel_lift_detected=False, back_rounded=False):
    # 1. Run the existing ML model (The "Base Score")
    features = np.array([[back_angle, back_angle, hip_angle, knee_angle, knee_angle]])
    prediction_encoded = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = float(np.max(probability))
    label = le.inverse_transform([prediction_encoded])[0]

    # 2. Base Scoring
    if label == "good":
        score = int(50 + confidence * 50)
    else:
        score = int((1 - confidence) * 50)

    # 3. THE WOW FACTOR: Safety Overrides
    # Even if the ML says "Good," these rules can override it
    reasons = []
    
    if heel_lift_detected:
        score -= 25
        reasons.append("HEELS LIFTED")
        
    if back_rounded:
        score -= 20
        reasons.append("BACK ROUNDED")

    # Final logic: If safety is compromised, it can't be "Good"
    if score < 60:
        label = "bad"
        
    # Ensure score stays between 0-100
    score = max(0, min(100, score))

    return label, score, confidence, reasons

#test
if __name__ == "__main__":
    label, score = predict_form(88, 92, 60)
    print(f"Label: {label}, Score: {score}/100")