import numpy as np
import cma

from switch_2_rod import run_sim

def objective(x):
    radius, h, p1x, p1y, p3x, p3y, p4x, p4y = x
    
    score = run_sim(radius, h, p1x, p1y, p3x, p3y, p4x, p4y, False)
    return score

x0 = np.array([10, 10.0, -50.0, 0.0, 50.0, 0.0, -70.0, -8.0])
sigma0 = 1.0
es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize":16, "maxiter":50})

while not es.stop():
    X = es.ask()
    F = [objective(x) for x in X]
    es.tell(X, F)
    es.disp()

best = es.result.xbest
print("best x:", best)
print("best fitness:", es.result.fbest)