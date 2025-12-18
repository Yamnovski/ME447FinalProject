import numpy as np
import cma
import matplotlib.pyplot as plt

from switch_sim import run_sim
from switch_sim_buckling import run_sim as run_sim_buckling

# r1, r2, r3, rod_4_stiff, preload, p1x, p1y, p3x, p3y, p4x, p4y
x = np.array([5.95077, 5.43655, 9.37571, 8.0, 4.0, -54.17448, -5.82995, 54.31005, -21.82463, -77.13239, -15.56851])
orientation_boundary_condition = False
score = run_sim_buckling(*x, orientation_boundary_condition, do_plots=True)