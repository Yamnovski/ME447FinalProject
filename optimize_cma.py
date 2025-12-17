import numpy as np
import cma

from switch_2_rod import run_sim

def objective(x):
    r1, r2, r3, p1x, p1y, p3x, p3y, p4x, p4y = x
    
    score = run_sim(r1, r2, r3, p1x, p1y, p3x, p3y, p4x, p4y, False)
    return score

x0 = np.array([6.0, 6.0, 6.0, -50.0, -10.0, 50.0, -10.0, -70.0, -18.0])
sigma0 = 3.0
es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize":10, "maxiter":5})


while not es.stop():
    X = es.ask()
    F = [objective(x) for x in X]
    es.tell(X, F)
    es.disp()

    best = es.result.xbest
    print("best x:", best)
    print("best fitness:", es.result.fbest)