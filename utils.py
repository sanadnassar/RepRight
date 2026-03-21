import numpy as np

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