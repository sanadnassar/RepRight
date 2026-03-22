import joblib
import numpy as np
import os

# Load once when file is imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "..", "machine_learning", "squat_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "machine_learning", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
le    = joblib.load(ENCODER_PATH)

def predict_form(knee_angle, hip_angle, back_angle, 
                 heel_lifted=False, back_rounded=False, 
                 knee_caving=False, lack_of_depth=False):
    """
    Combines Machine Learning (angles) with Biomechanical Hard-Rules (flags).
    """
    
    # 1. THE ML BASE: Mirror the exact feature order Aref used
    features = np.array([[
        back_angle,   # torso lean
        back_angle,   
        hip_angle,    # hip hinge
        knee_angle,   # knee flexion
        knee_angle    
    ]])

    prediction_encoded = model.predict(features)[0]
    probability        = model.predict_proba(features)[0]
    confidence         = float(np.max(probability))
    label              = le.inverse_transform([prediction_encoded])[0]

    # 2. BASE SCORING
    if label == "good":
        score = int(50 + confidence * 50)   # Range: 50-100
    else:
        score = int((1 - confidence) * 50)  # Range: 0-50

    # 3. HEURISTIC OVERRIDES: Penalizing 'The Bad Stuff'
    reasons = []
    
    if heel_lifted:
        score -= 30
        reasons.append("HEELS LIFTED")
        
    if knee_caving:
        score -= 25
        reasons.append("KNEES CAVING")
        
    if back_rounded:
        score -= 20
        reasons.append("BACK ROUNDED")

    if lack_of_depth and knee_angle > 105:
        score -= 15
        reasons.append("LOW DEPTH")

    # 4. FINAL LABEL RE-EVALUATION
    # If the penalties drag the score down, the form is no longer "good"
    

    # Clamp score between 0 and 100
    final_score = max(0, min(100, score))

    return label, final_score, confidence, reasons