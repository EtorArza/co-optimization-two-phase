from argparse import ArgumentError

import subprocess
import time
import re
from os.path import exists
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse



def launch_one(experiment_index):
    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("gymrem2d", experiment_index)
    params.print_parameters()

    no = NestedOptimization("results/gymrem2d/data", params)
    sys.path.append(sys.path[0]+"/../other_repos/gymrem2d/ModularER_2D/")
    print(sys.path)
    import REM2D_main
    from REM2D_main import setup, run2D
    config, dir = setup(no)
    experiment = run2D(no,config,dir)
    experiment.run(config)



if sys.argv[1] == "--local_launch":
    import itertools
    import time




    launch_one(int(sys.argv[2]))





    # def run_with_seed(seed):

    #     time.sleep(0.5)
    #     print(f"Launching with seed {seed} in experiment_halveruntime.py ...")

    #     bash_cmd = f"python3 other_RL/gym_rem2D/ModularER_2D/Demo2_Evolutionary_Run.py --method {method} --seed {seed} --gracetime {gracetime} --res_filepath {res_filepath}"
    #     print(bash_cmd)
    #     exec_res=subprocess.run(bash_cmd,shell=True, capture_output=True)
        
    # #     run_with_seed_and_runtime(seed, "halving")
    # Parallel(n_jobs=parallel_threads, verbose=12)(delayed(run_with_seed)(i) for i in seeds)


