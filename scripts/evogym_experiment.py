import sys
sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os

def new_argparse():
    print("Yolo")
    exit(1)





if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    os.chdir("other_repos/evogym/examples")
    run_ga(
        max_evaluations = 250,
        train_iters = 1000,
        num_steps = 128,
        env_name = "Walker-v0",
        pop_size = 25,
        structure_shape = (5,5),
        experiment_name = "test_ga",
        num_cores = 1,
    )
    os.chdir("../../..")



    # python run_ga.py --env-name "Walker-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 50

