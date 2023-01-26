import sys
sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os
from NestedOptimization import NestedOptimization
from multiprocessing.managers import BaseManager
from plot_src import *

figpath = "results/evogym/figures"

def get_sequence_of_parameters():
    seed_list = list(range(2,22))
    inners_per_outer_list = [1000, 500] # Default is 1000
    inner_length_proportion_list = [1.0, 0.5] # Default is 1.0
    return list(itertools.product(seed_list, inners_per_outer_list, inner_length_proportion_list))





if __name__ == "__main__":
    if sys.argv[1] == "--first_iteration":
        sys.argv.pop()
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        os.chdir("other_repos/evogym/examples")
        BaseManager.register('NestedOptimization', NestedOptimization)
        manager = BaseManager()
        manager.start()
        # no = manager.NestedOptimization("../../../../../../../Documents/results_08/result_all.txt", "saveall")
        no = manager.NestedOptimization("../../../results/evogym/data/first_iteration.txt", "saveall")
        run_ga(
            experiment_name = "first_iteration",
            env_name = "Walker-v0",
            seed = 2,
            max_evaluations = 4, # Number of morphologies evaluated
            train_iters = 1000,    # Number of iterations for training each morphology
            num_steps = 128,       # Number of steps in each iteration
            pop_size = 4,          # Population size of the morphologies
            structure_shape = (5,5),
            num_cores = 1,
            no = no,
        )
        os.chdir("../../..")


    elif sys.argv[1] == "--plot":
        import pandas as pd
        from matplotlib import pyplot as plt
        df = pd.read_csv("results/evogym/data/first_iteration.txt")
        print("Inner learning algorithm in evogym is ppo.")
        plot_first_iteration(df, figpath, "evogym")
    
    else:
        ValueError("sys.argv[1] was ", sys.argv[1], " and this is not a recognized experiment.")
  