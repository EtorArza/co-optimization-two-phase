import sys
sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os
from NestedOptimization import NestedOptimization
from multiprocessing.managers import BaseManager
import itertools
from plot_src import *

figpath = "results/evogym/figures"

def get_sequence_of_parameters():
    seed_list = list(range(2,22))
    inners_per_outer_proportion_list = [1.0, 0.5] # Default is 1000
    inner_length_proportion_list = [1.0, 0.5] # Default is 64
    return list(itertools.product(seed_list, inners_per_outer_proportion_list, inner_length_proportion_list))

def execute_experiment_locally(seed, max_frames, inners_per_outer_proportion, inner_length_proportion):
    random.seed(seed)
    np.random.seed(seed)
    os.chdir("other_repos/evogym/examples")
    mode = ['saveall','standard'][1]

    # # Parallel
    # BaseManager.register('NestedOptimization', NestedOptimization)
    # manager = BaseManager()
    # manager.start()
    # no = manager.NestedOptimization(f"../../../results/evogym/data/{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}.txt", mode, max_frames, inners_per_outer_proportion, inner_length_proportion)

    # Sequential
    no = NestedOptimization(f"../../../results/evogym/data/{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}.txt", mode, max_frames, inners_per_outer_proportion, inner_length_proportion)
    env_name = "Walker-v0"

    run_ga(
        experiment_name = f"{env_name}_{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}",
        env_name = env_name,
        seed = seed,
        max_evaluations = 20000,
        pop_size = 25,
        structure_shape = (5,5),
        num_cores = 1,
        no = no,
    )
    os.chdir("../../..")



if __name__ == "__main__":
    if sys.argv[1] == "--local_launch":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --local_launch and i.\n\nUsage:\npython src/robogrammar_experiment.py i")
            exit(1)
        i = int(sys.argv[2])
        sys.argv = sys.argv[:1]
        seq_parameters = get_sequence_of_parameters()
        print("Total number of executions:", len(seq_parameters))
        print("Parameters current execution:",seq_parameters[i])
        seed, inners_per_outer_proportion, inner_length_proportion = seq_parameters[i]
        # max_frames=32032000 is the default value if we consider 250 morphologies evaluated.
        execute_experiment_locally(seed=seed, max_frames=384384, inners_per_outer_proportion=inners_per_outer_proportion, inner_length_proportion=inner_length_proportion)



    elif sys.argv[1] == "--plot":
        import pandas as pd
        from matplotlib import pyplot as plt
        df = pd.read_csv("results/evogym/data/first_iteration.txt")
        print("Inner learning algorithm in evogym is ppo.")
        plot_first_iteration(df, figpath, "evogym")
    
    else:
        ValueError("sys.argv[1] was ", sys.argv[1], " and this is not a recognized experiment.")
  