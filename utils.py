import numpy as np
KNEE_DEPTH_THRESHOLD  = 100
BACK_LEAN_THRESHOLD   = 70 
HIP_HINGE_THRESHOLD   = 110



# Calculates angle to help with rating form
def calculate_angle(a, b, c):
    # Calculate input coordinates into NumPy arrays for vector operations
    a, b, c = np.array(a), np.array(b), np.array(c)

    # Create vectors originating from the vertex 'b'
    ba, bc = a - b, c - b

    # Calculate the cosine of the angle using the Dot Product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clip the cosine value to the range [-1.0, 1.0] to prevent NaN errors
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

def get_feedback(knee_angle, hip_angle, back_angle, label):
    feedback = []

    # knee angle feedback
    if knee_angle < 70:
        feedback.append("Excellent depth — below parallel")
    elif knee_angle < 90:
        feedback.append("Good depth — just at parallel")
    elif knee_angle < 100:
        feedback.append("Slightly shallow — push for parallel")
    else:
        feedback.append("Too shallow — drive knees out and sit deeper")

    #back angle feedback
    if back_angle < 45:
        feedback.append("Back too upright — slight forward lean is normal")
    elif back_angle < 60:
        feedback.append("Torso angle is ideal")
    elif back_angle < 70:
        feedback.append("Slight forward lean — brace your core")
    else:
        feedback.append("Excessive forward lean — risk of injury")

    # hip feedback
    if hip_angle < 85:
        feedback.append("Good hip hinge depth")
    elif hip_angle < 110:
        feedback.append("Hips need to drop lower")
    else:
        feedback.append("Hips too high — this is a good morning, not a squat")

    # overall verdict
    if label == "good":
        verdict = "Solid rep"
    else:
        verdict = "Needs work"

    return verdict + " | " + " | ".join(feedback)