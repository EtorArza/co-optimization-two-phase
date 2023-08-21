import os
import sys

if sys.argv[1] == "--clean":
    import os
    os.system("rm -f gymrem2d_local_launch.txt")
    os.system("rm -f gymrem2d_local_launch.txt_log.txt")
    os.system("rm -f gymrem2d_tune.txt")
    os.system("rm -f gymrem2d_tune.txt_log.txt")
    os.system("rm -f results/gymrem2d/data/meth*")
    os.system("rm -f results/gymrem2d/figures/meth*")
    os.system("rm -f results/gymrem2d/videos/meth*")
    os.system("rm -f other_repos/gymrem2d/dumps_for_animation/anim*")
    print("All experiments deleted.")
    exit(0)


from argparse import ArgumentError

import subprocess
import time
import re
from os.path import exists
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse
from plot_src import *



def launch_one(experiment_index):
    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("gymrem2d", experiment_index)
    params.print_parameters()
    no = NestedOptimization("../../../results/gymrem2d/data", params)
    sys.path.append(sys.path[0]+"/../other_repos/gymrem2d/ModularER_2D/")
    print(sys.path)
    import REM2D_main
    from REM2D_main import setup, run2D
    config, dir = setup(no)
    experiment = run2D(no,config,dir)
    experiment.run(config)


def launch_one_parameter_tuning(seed, default_inner_quantity):
    from NestedOptimization import Parameters, NestedOptimization
    params = Parameters("gymrem2d", 1)
    params.seed = seed
    def return_filename():
        return f"paramtuning_{default_inner_quantity}_{seed}"
    params.get_result_file_name=return_filename
    params._default_inner_quantity = default_inner_quantity
    params._default_inner_length = 360
    params._inner_quantity_proportion = 1.0 
    params._inner_length_proportion = 1.0

    params.print_parameters()
    no = NestedOptimization("../../../results/gymrem2d/data", params, deletePreviousResults=True, limit_the_amount_of_written_lines=True if default_inner_quantity == 1 else False)
    sys.path.append(sys.path[0]+"/../other_repos/gymrem2d/ModularER_2D/")
    print(sys.path)
    import REM2D_main
    from REM2D_main import setup, run2D
    REM2D_main.save_data_animation = lambda dump_path, no, ind, tree_dpth, video_label: None
    

    config, dir = setup(no)
    experiment = run2D(no,config,dir)
    experiment.run(config)


    

if sys.argv[1] == "--local_launch":
    import itertools
    import time
    launch_one(int(sys.argv[2]))

elif sys.argv[1] == "--local_launch_tuning":
    import itertools
    import time
    assert len(sys.argv) == 4
    launch_one_parameter_tuning(int(sys.argv[2]), int(sys.argv[3]))

elif sys.argv[1] == "--visualize":
    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("gymrem2d", int(sys.argv[2]))
    params.print_parameters()

    no = NestedOptimization("../../../results/gymrem2d/data", params)
    sys.path.append(sys.path[0]+"/../other_repos/gymrem2d/ModularER_2D/")
    print(sys.path)
    import REM2D_main
    from REM2D_main import animate_from_dump

    animate_from_dump(f"other_repos/gymrem2d/dumps_for_animation/animation_dump_current{int(sys.argv[2])}.wb")
    animate_from_dump(f"other_repos/gymrem2d/dumps_for_animation/animation_dump_best{int(sys.argv[2])}.wb")

elif sys.argv[1] == "--visualize_all":
    for i, method in zip([8, 17, 4, 13, 0, 9], ["standard"]*2+["reduced_length"]*2 + ["reduced_quantity"]*2):
        os.system(f"python src/gymrem2d_experiment.py --visualize {i}")
        os.system(f"mv results/gymrem2d/videos/vid_reevaleachvsend_{i}_*_best.gif animations_for_the_paper/tiny_quantity_vs_length_gymrem2d/gif_{method}_{i}_gymrem2d.gif")
    os.system("rm results/gymrem2d/videos/vid_reevaleachvsend_*.gif")

elif sys.argv[1] == "--tune":
    seeds = list(range(40))
    from itertools import product
    from NestedOptimization import convert_from_seconds, experimentProgressTracker
    import joblib

    threads = 8
    parameter_combs = list(product(seeds, [1, 8, 16, 32, 64, 128, 512]))
    progress_filename = "gymrem2d_tune.txt"
    start_index = 0
    prog = experimentProgressTracker(progress_filename, 0, len(parameter_combs))
    
    def launch_next(prog: experimentProgressTracker):
        i = prog.get_next_index()
        seed, default_inner_quantity = parameter_combs[i]

        exit_status = os.system(f"python src/gymrem2d_experiment.py --local_launch_tuning {seed} {default_inner_quantity}")
        if exit_status == 0:
            prog.mark_index_done(i)
        else:
            print(exit_status)
            exit(1)

    Parallel(n_jobs=threads)(delayed(launch_next)(prog) for _ in range(len(parameter_combs)))


elif sys.argv[1] == "--local_launch_all":
    
    from itertools import product
    import joblib
    from NestedOptimization import Parameters, NestedOptimization, experimentProgressTracker
    params = Parameters("gymrem2d", 1)
    n = params.get_n_experiments()
    n_visualize = 21
    threads = 8
    prog = experimentProgressTracker("gymrem2d_local_launch.txt",0,n)

    def launch_next(prog: experimentProgressTracker):
        i = prog.get_next_index()
        exit_status = os.system(f"python src/gymrem2d_experiment.py --local_launch {i}")
        if exit_status != 0:
            print(exit_status)
            exit(1)
        else:
            prog.mark_index_done(i)

    Parallel(n_jobs=threads)(delayed(launch_next)(prog) for _ in range(n))

    for i in range(n_visualize):
        os.system(f"python src/gymrem2d_experiment.py --visualize {i}")




elif sys.argv[1] == "--get_frames_by_default":
    import numpy as np
    from matplotlib import pyplot as plt


    def get_n_frames_and_evals(seed):
        with open(f"/home/paran/Dropbox/BCAM/07_estancia_1/code/results/data/veenstra/problemspecific_{seed}.txt", "r") as f:
            lines = f.readlines()
            while len(lines[-1]) < 50:
                line = lines.pop()
            line = lines.pop()
            frames = int(line.split(",")[3])
            evals = (int(line.split(",")[4]) + 1) * 100
            frames_per_episode_last_gen = (frames - int(lines.pop().split(",")[3])) / 100
            return frames, evals, frames_per_episode_last_gen


    frames = np.array([get_n_frames_and_evals(seed)[0] for seed in range(2,23)])
    frames_per_eval = np.array([get_n_frames_and_evals(seed)[0] / get_n_frames_and_evals(seed)[1] for seed in range(2,23)])
    frames_per_episode_last_gen = np.array([get_n_frames_and_evals(seed)[2] for seed in range(2,23)])

    print("On average, the default gymrem2d experiment uses", np.mean(frames), "frames in total.")
    print("On average, the default gymrem2d experiment uses", np.mean(frames_per_eval), "frames per evaluated controller (average episode length in frames).")
    print("On average, IN THE LAST GENERATION, the default gymrem2d experiment uses", np.mean(frames_per_episode_last_gen), "frames per evaluated controller (average episode length in frames).")

elif sys.argv[1] == "--plot":
    df = plot_comparison_parameters("results/gymrem2d/data", "results/gymrem2d/figures")

elif sys.argv[1] == "--plot_tune":
    plot_tune("results/gymrem2d/data", "results/gymrem2d/figures")


else:
    raise ValueError(f"Argument {sys.argv[1]} not recognized.")