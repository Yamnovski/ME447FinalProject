import numpy as np
import cma
import matplotlib.pyplot as plt

from switch_sim import run_sim
from switch_sim_buckling import run_sim as run_sim_buckling

# r1, r2, r3, rod_4_stiff, preload, p1x, p1y, p3x, p3y, p4x, p4y
x = np.array([5.00333, 5.35735, 6.5223, 6.5, 2.37472, -53.41759, -7.60989, 62.23931, -18.36738, -83.59239, -17.96934])
orientation_boundary_condition = False
score = run_sim_buckling(*x, orientation_boundary_condition, do_plots=True)

# r1, r2, r3, rod_4_stiff, preload, p1x, p1y, p3x, p3y, p4x, p4y
5.00333, 5.35735, 6.5223, 4.0, 2.37472, -53.41759, -10.60989, 62.23931, -22.36738, -83.59239, -17.96934