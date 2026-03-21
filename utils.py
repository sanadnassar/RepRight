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
    if label == "good":
        return "Good depth! Solid form."
    
    #feedback for bad form
    feedback = []
    
    if knee_angle > KNEE_DEPTH_THRESHOLD:
        feedback.append("Go deeper — knee angle too wide")
    if back_angle > BACK_LEAN_THRESHOLD:
        feedback.append("Stay upright — too much forward lean")
    if hip_angle > HIP_HINGE_THRESHOLD :
        feedback.append("Drive hips back further")
    
    return " | ".join(feedback) if feedback else "Bad form detected"