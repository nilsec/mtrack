import numpy as np
import math

def get_spanning_angle(vec_1, vec_2):
    v1 = np.array(vec_1)
    v2 = np.array(vec_2)
    assert len(v1) == 3
    assert len(v2) == 3

    if sum(v1) == 0 or sum(v2) == 0:
        return 0
    elif (v1 == v2).all():
        return 0
    elif ((v1 * (-1)) == v2).all():
        # print "zero or 180 degree"
        return np.pi
    else:
        try:
            angle = math.acos(np.dot(v1, v2) / ((np.linalg.norm(v1)) * np.linalg.norm(v2)))
            return angle
        except ValueError:
            print "value close to zero"
            #     print v1, v2
            angle = 0
    return angle

def get_orientation_angle(v_positions, v_orientations):
    assert(len(v_positions) == 2)
    assert(len(v_orientations) == 2)

    vector = v_positions[0] - v_positions[1]
    orientation_angle = []

    for orientation in v_orientations:
        angle = get_spanning_angle(vector, orientation)
        
        if angle >= np.pi/2.:
            assert(angle <= np.pi)
            angle = np.pi - angle

        orientation_angle.append(angle)

    return orientation_angle
