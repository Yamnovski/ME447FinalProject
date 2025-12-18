import numpy as np
import cma
import matplotlib.pyplot as plt

from switch_sim import run_sim
from switch_sim_buckling import run_sim as run_sim_buckling
from concurrent.futures import ProcessPoolExecutor

import os
import tempfile
import shutil

MAIN_DIR = os.getcwd() 

orientation_boundary_condition = False

def objective(x):
    r1, r2, r3, rod_4_stiff, preload, p1x, p1y, p3x, p3y, p4x, p4y = x
    
    worker_dir = tempfile.mkdtemp(prefix="worker_") # create a temp folder for each process/worker
    data_file = os.path.join(MAIN_DIR, "target_force_data.csv") 
    shutil.copy(data_file, worker_dir) # copy the target_force_data.csv file into each folder
    os.chdir(worker_dir) # change the working directory for the rest of the objective function call to the temp folder

    try:
        score = run_sim_buckling(r1, r2, r3, rod_4_stiff, preload, 
                        p1x, p1y, p3x, p3y, p4x, p4y, orientation_boundary_condition, False)
    except Exception as e:
        score = 999999999999999
        print(e)
    return score


x0 = np.array([5.95077, 5.43655, 9.37571, 4.0, 0.5, -54.17448, -5.82995, 54.31005, -21.82463, -77.13239, -15.56851])
sigma0 = 0.5
popsize = 10
es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize":popsize, "maxiter":40})
num_cores = os.cpu_count()  # e.g., 8
num_processes = min(popsize, num_cores)

best_fitness_arr = []
best_soln_arr = []


while not es.stop():
    X = es.ask()
    # F = [objective(x) for x in X]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # F = list(executor.map(objective, X))
        F = list(executor.map(objective, X))
    
    es.tell(X, F)
    es.disp()

    best = es.result.xbest
    print("best x:", best)
    print("best fitness:", es.result.fbest)
    best_fitness_arr.append(es.result.fbest)
    best_soln_arr.append(best)

    fig = plt.figure()
    plt.plot(best_fitness_arr)
    plt.title("Fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    # plt.legend("Best solution: " + " ".join(str(param) for param in best))
    plt.savefig("plot_fitness.png", bbox_inches='tight')
    # plt.show()
    plt.close(fig)



