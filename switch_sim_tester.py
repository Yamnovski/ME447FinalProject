import numpy as np
import cma
import matplotlib.pyplot as plt

from switch_sim import run_sim


# r1, r2, r3, p1x, p1y, p3x, p3y, p4x, p4y
# x = np.array([4.94772, 5.24589, 5.05957, -67.0155, -0.233837, 54.0787, -30.9179, -73.6563, -0.0556065])
x = np.array([5.95077, 5.43655, 9.37571, -54.17448, -5.82995, 54.31005, -21.82463, -77.13239, -15.56851])
orientation_boundary_condition = False
score = run_sim(*x, orientation_boundary_condition, True)