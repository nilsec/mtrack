import angles
import numpy as np
from time import time

a = np.random.randn(10**6, 3)
b = np.random.randn(10**6, 3)

t1 = time()

for v1, v2 in zip(a,b):
    angles.get_spanning_angle(v1, v2)

t2 = time()

print t2 - t1

v_positions = [[np.random.randn(3), np.random.randn(3)] for j in range(10**6)]
v_orientations = [[np.random.randn(3), np.random.randn(3)] for j in range(10**6)]

t1 = time()

for p, o in zip(v_positions, v_orientations):
    angles.get_orientation_angle(p, o)

t2 = time()

print t2 - t1

