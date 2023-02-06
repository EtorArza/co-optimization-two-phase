import sys
import itertools
figpath = "results/robogrammar/figures"



def get_sequence_of_parameters():
    seed_list = list(range(2,22))
    inners_per_outer_proportion_list = [1.0, 0.5] # Default is 64
    inner_length_proportion_list = [1.0, 0.5] # Default is 128
    return list(itertools.product(seed_list, inners_per_outer_proportion_list, inner_length_proportion_list))

def execute_experiment_locally(seed, max_frames, inners_per_outer_proportion, inner_length_proportion):
    if sys.executable.split('/')[-3] != 'venv':
        print("This script requires that conda is deactivated and the python environment in other_repos/RoboGrammar/venv/bin/activate is activated. To achieve this, run the following: \n\nconda deactivate\nsource other_repos/RoboGrammar/venv/bin/activate")
        print("\n\nOnce 'venv' has been loaded, rerun this script.")
        exit(1)

    import torch
    import os

    os.chdir("other_repos/RoboGrammar")
    # sys.path.append("./other_repos/RoboGrammar/examples/graph_learning")
    # from heuristic_search_algo_mpc import *
    sys.path.append("./other_repos/RoboGrammar/examples/design_search")
    from design_search import main

    sys.argv.pop()
    sys.argv.pop()
    mode = ['saveall','standard'][1]
    algorithm = ["mcts", "random"][0]
    cpus = 1
    task = 'FlatTerrainTask'
    resfilepath = f"../../results/robogrammar/data/{task}_{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}_{seed}.txt"


    torch.set_default_dtype(torch.float64)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'graph_learning'))
    sys.path.append(os.path.join(base_dir, 'design_search'))
    from NestedOptimization import NestedOptimization
    import os
    no = NestedOptimization(resfilepath, mode, max_frames, inners_per_outer_proportion, inner_length_proportion)
    main(no, algorithm, cpus, task, seed)


if __name__ == "__main__":


    if sys.argv[1] == "--local_launch":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --local_launch and i.\n\nUsage:\npython src/robogrammar_experiment.py i")
            exit(1)
        i = int(sys.argv[2])
        seq_parameters = get_sequence_of_parameters()
        print("Number of executions:", len(seq_parameters))
        seed, inners_per_outer_proportion, inner_length_proportion = seq_parameters[i]
        # max_frames=262144000 is the default value if we consider only 2000 iterations in their paper.
        execute_experiment_locally(seed=seed, max_frames=262144000, inners_per_outer_proportion=inners_per_outer_proportion, inner_length_proportion=inner_length_proportion)

        
    elif sys.argv[1] == "--plot":
        from plot_src import *
        import pandas as pd
        from matplotlib import pyplot as plt
        print("Inner learning algorithm in evogym is MPC.")
        df = pd.read_csv("results/robogrammar/data/first_iteration.txt")
        plot_first_iteration(df, figpath, "RoboGrammar")


    # elif sys.argv[1] == "--cluster_launch":
        