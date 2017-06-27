import numpy as np
from numpy.linalg import norm
import math

def get_spanning_angle(vec_1, vec_2):
    assert vec_1.shape[0] == 3 and vec_2.shape[0] == 3

    norm_1 = math.sqrt(vec_1[0]**2 + vec_1[1]**2 + vec_1[2]**2) 
    norm_2 = math.sqrt(vec_2[0]**2 + vec_2[1]**2 + vec_2[2]**2)

    u1 = vec_1/norm_1
    u2 = vec_2/norm_2

    angle = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return angle 
 
    
def get_orientation_angle(v_positions, v_orientations):
    assert(len(v_positions) == 2)
    assert(len(v_orientations) == 2)

    vector = v_positions[0] - v_positions[1]
    orientation_angle = np.empty(2)

    j = 0
    for orientation in v_orientations:
        angle = get_spanning_angle(vector, orientation)
        
        if angle >= np.pi/2.:
            assert(angle <= np.pi)
            angle = np.pi - angle

        orientation_angle[j] = angle
        j += 1

    return orientation_angle
