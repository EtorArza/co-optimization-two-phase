import sys
sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os
from NestedOptimization import NestedOptimization
from multiprocessing.managers import BaseManager

def new_argparse():
    print("Yolo")
    exit(1)





if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.chdir("other_repos/evogym/examples")
    BaseManager.register('NestedOptimization', NestedOptimization)
    manager = BaseManager()
    manager.start()
    no = manager.NestedOptimization("../../../results/evogym/data/test_results.txt")
    run_ga(
        experiment_name = "test_ga",
        env_name = "Walker-v0",
        seed = 2,
        max_evaluations = 250, # Number of morphologies evaluated
        train_iters = 100,    # Number of iterations for training each morphology
        num_steps = 128,       # Number of steps in each iteration
        pop_size = 4,          # Population size of the morphologies
        structure_shape = (5,5),
        num_cores = 5,
        no = no,
    )
    #7.792172 128128
    # 1.403425 128128
    # 2.734754 128128
    # 4.80972 128128
    os.chdir("../../..")



    # python run_ga.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50

