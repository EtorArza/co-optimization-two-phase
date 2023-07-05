import os
import sys

if sys.argv[1] == "--clean":
    import os
    os.system("rm -f gymrem2d_local_launch.txt")
    os.system("rm -f gymrem2d_local_launch.txt_log.txt")
    os.system("rm -f gymrem2d_tune.txt")
    os.system("rm -f gymrem2d_tune.txt_log.txt")
    os.system("rm -f results/gymrem2d/data/*")
    os.system("rm -f results/gymrem2d/figures/*")
    os.system("rm -f results/gymrem2d/videos/*")
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
    import os
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt

    def find_between(s, start, end): # find substring between two strings
        return (s.split(start))[1].split(end)[0]

    exp_dir = "results/gymrem2d/data"
    fig_dir = "results/gymrem2d/figures"

    rows = []
    for csv_name in os.listdir(exp_dir):
        if ".txt" in csv_name and "paramtuning" in csv_name:
            df = pd.read_csv(exp_dir + "/" + csv_name)
            f = df.query("level == 2")["f_best"].iloc[-1]
            step = df.query("level == 2")["step"].iloc[-1]
            nrows = df.query("level == 2").shape[0]
            innerquantity = int(find_between(csv_name, "paramtuning_","_"))
            seed = int(find_between(csv_name, "_",".txt"))
            rows.append([innerquantity, seed, f, nrows, step])
    df = pd.DataFrame(rows, columns=["innerquantity", "seed", "f", "nrows","step"])

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    inner_quantity_list = sorted(df["innerquantity"].unique())
    print("inner_quantity_list =", inner_quantity_list)

    # # https://stackoverflow.com/questions/43345599/process-pandas-dataframe-into-violinplot
    # fig, axes = plt.subplots()
    # axes.violinplot(dataset = [df[df.innerquantity == el]["f"].values for el in inner_quantity_list],showmedians=True)
    # axes.set_title('Day Ahead Market')
    # axes.yaxis.grid(True)
    # axes.set_xlabel('Scenario')
    # axes.set_ylabel('LMP ($/MWh)')
    # plt.show()
    # plt.close()

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Controllers evaluated per morphology')

    boxplot = plt.boxplot([df[df.innerquantity == el]["f"].values for el in inner_quantity_list], showmeans=True)

    # Add legend handles and labels
    legend_handles = [boxplot["medians"][0], boxplot["means"][0]]
    legend_labels = ["Median", "Mean"]
    plt.legend(legend_handles, legend_labels)

    set_axis_style(plt.gca(), [str(el) for el in inner_quantity_list])
    plt.title("f")
    plt.savefig(fig_dir+r"/f_tune.pdf")
    plt.close()

    plt.violinplot(dataset = [df[df.innerquantity == el]["nrows"].values for el in inner_quantity_list],showmedians=True)
    set_axis_style(plt.gca(), [str(el) for el in inner_quantity_list])
    plt.title("nrows")
    plt.yscale("log")
    plt.savefig(fig_dir+r"/nrows_tune.pdf")
    plt.close()

    plt.violinplot(dataset = [df[df.innerquantity == el]["step"].values for el in inner_quantity_list],showmedians=True)
    set_axis_style(plt.gca(), [str(el) for el in inner_quantity_list])
    plt.title("step")
    plt.yscale("log")
    plt.savefig(fig_dir+r"/step_tune.pdf")
    plt.close()


    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 6))


    markers = ['o', 's', 'D', 'v', '^', 'p', 'P']
    line_styles = ['-', '--', '-.', ':']
    line_colors = ['black', 'gray', 'darkgray', 'lightgray']
    num_colors = len(inner_quantity_list)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Greys', num_colors+1)  # Linear colormap from light to dark
    marker_step = 10  # Set the step size for markers

    for idx, inner_quantity in enumerate(inner_quantity_list):
        data = df[df.innerquantity == inner_quantity]["f"].values
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 100)
        density = kde(x)
        cumulative_density = np.cumsum(density) / np.sum(density)  # Compute cumulative distribution
        color = cmap(idx+1)  # Get color from the colormap
        ax.plot(x, cumulative_density, marker=markers[idx], linestyle=line_styles[idx % len(line_styles)],
                color=color, markevery=marker_step, label=str(inner_quantity))

    ax.set_xlabel('Objective value')
    ax.set_ylabel('Cumulative Distribution')
    ax.set_title('f_cumulative')
    ax.legend()

    plt.savefig(fig_dir + r"/f_tune_cumulative.pdf")
    plt.show()
    plt.close()


else:
    raise ValueError(f"Argument {sys.argv[1]} not recognized.")