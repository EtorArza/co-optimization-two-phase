import sys
sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os
from NestedOptimization import NestedOptimization
from multiprocessing.managers import BaseManager

figpath = "results/evogym/figures"




if sys.argv[1] == "--first_iteration":
    sys.argv.pop()
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.chdir("other_repos/evogym/examples")
    BaseManager.register('NestedOptimization', NestedOptimization)
    manager = BaseManager()
    manager.start()
    # no = manager.NestedOptimization("../../../../../../../Documents/results_08/result_all.txt", "save_all")
    no = manager.NestedOptimization("../../../results/evogym/data/first_iteration.txt", "save_all")
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
    #7.792172 128128
    # 1.403425 128128
    # 2.734754 128128
    # 4.80972 128128
    os.chdir("../../..")


elif sys.argv[1] == "--plot":
    import pandas as pd
    from matplotlib import pyplot as plt
    df = pd.read_csv("results/evogym/data/first_iteration.txt")

    print("Plot fitness in first 30 iterations in ppo on evogym...", end="") 
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sub_df = df.query("evaluation == 0 & iteration < 30 & level == 0").groupby("iteration")["f"].plot()
    plt.yscale("symlog")
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title("ppo first 30 iterations evogym")
    plt.tight_layout()
    plt.savefig(figpath + "/ppo_first_30_iterations.pdf")
    plt.close()
    print("done")




    print("Plot fitness first 4 evaluations in ppo on evogym...", end="")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    grouped = df.query("level == 1 & f.notna()").groupby("evaluation")
    for key in grouped.groups.keys():
        plt.plot(grouped.get_group(key)["iteration"], grouped.get_group(key)["f"])
    plt.xlabel("iteration")
    plt.ylabel("objective value")
    plt.title("Evaluation of first four morphologies")
    plt.tight_layout()
    plt.savefig(figpath+"/first_4_evaluations.pdf")
    plt.close()
    print("done")
 

    # python run_ga.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50

