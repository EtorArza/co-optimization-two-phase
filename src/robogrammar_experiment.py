import sys
sys.path.append("./other_repos/RoboGrammar/examples/graph_learning")
sys.path.append("./other_repos/RoboGrammar/examples/design_search")
from heuristic_search_algo_mpc import *
from design_search import *
figpath = "results/robogrammar/figures"


def execute_experiment_locally(seed, max_frames, inners_per_outer, inner_length_proportion):
    if sys.executable.split('/')[-3] != 'venv':
        print("This script requires that conda is deactivated and the python environment in other_repos/RoboGrammar/venv/bin/activate is activated. To achieve this, run the following: \n\nconda deactivate\nsource other_repos/RoboGrammar/venv/bin/activate")
        print("\n\nOnce 'venv' has been loaded, rerun this script.")
        exit(1)

    import torch
    import os

    sys.argv.pop()
    mode = ['saveall','standard'][1]
    algorithm = ["mcts", "random"][0]
    cpus = 6
    it_params = {
    "max_frames": max_frames,
    "inners_per_outer": inners_per_outer,
    "inner_length_proportion":inner_length_proportion,
    }
    task = 'FlatTerrainTask'
    resfilepath = f"../../results/robogrammar/data/{max_frames}_{inners_per_outer}_{inner_length_proportion}.txt"
    os.chdir("other_repos/RoboGrammar")


    torch.set_default_dtype(torch.float64)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'graph_learning'))
    sys.path.append(os.path.join(base_dir, 'design_search'))
    from NestedOptimization import NestedOptimization
    import os
    no = NestedOptimization(resfilepath, mode, it_params)
    main(no, algorithm, cpus, task, seed)


if __name__ == "__main__":


    if sys.argv[1] == "--first_iteration":
        execute_experiment_locally(seed=2, max_frames=262144000, inners_per_outer=64, inner_length_proportion=1.0)

        
    elif sys.argv[1] == "--plot":
        import evogym_experiment
        import pandas as pd
        from matplotlib import pyplot as plt
        print("Inner learning algorithm in evogym is MPC.")
        df = pd.read_csv("results/robogrammar/data/first_iteration.txt")
        evogym_experiment.plot_first_iteration(df, figpath, "RoboGrammar")

