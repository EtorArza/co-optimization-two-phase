import sys
import os

figpath = "results/robogrammar/figures"


if __name__ == "__main__":


    if sys.argv[1] == "--first_iteration":

        if sys.executable.split('/')[-3] != 'venv':
            print("This script requires that conda is deactivated and the python environment in other_repos/RoboGrammar/venv/bin/activate is activated. To achieve this, run the following: \n\nconda deactivate\nsource other_repos/RoboGrammar/venv/bin/activate")
            print("\n\nOnce 'venv' has been loaded, rerun this script.")
            exit(1)

        import torch
        sys.path.append("./other_repos/RoboGrammar/examples/graph_learning")
        sys.path.append("./other_repos/RoboGrammar/examples/design_search")
        from heuristic_search_algo_mpc import *
        from design_search import *

        sys.argv.pop()
        seed = 2
        mode = ['saveall','standard'][0]
        algorithm = ["mcts", "random"][0]
        cpus = 1
        iterations = 4 # 2000
        task = 'FlatTerrainTask'
        resfilepath = "../../results/robogrammar/data/first_iteration.txt"
        os.chdir("other_repos/RoboGrammar")


        torch.set_default_dtype(torch.float64)
        # args_list = ['--grammar-file', 'data/designs/grammar_apr30.dot',
        #             '--num-samples', '1', 
        #             '--opt-iter', '25', 
        #             '--batch-size', '32',
        #             '--states-pool-capacity', '10000000',
        #             '--depth', '10',
        #             '--max-nodes', '20',
        #             '--save-dir', './trained_models/',
        #             '--log-interval', '100',
        #             '--task', 'FlatTerrainTask',
        #             '--eval-interval', '1000',
        #             '--max-trials', '1000',
        #             '--num-eval', '1',
        #             '--num-iterations', '4',
        #             '--mpc-num-processes', '4',
        #             '--seed', str(seed),
        #             '--no-noise',
        #             ]

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        sys.path.append(base_dir)
        sys.path.append(os.path.join(base_dir, 'graph_learning'))
        sys.path.append(os.path.join(base_dir, 'design_search'))
        from NestedOptimization import NestedOptimization
        import os
        no = NestedOptimization(resfilepath, mode)
        main(no, algorithm, cpus, iterations, task, seed)
        
    elif sys.argv[1] == "--plot":
        import evogym_experiment
        import pandas as pd
        from matplotlib import pyplot as plt
        print("Inner learning algorithm in evogym is MPC.")
        df = pd.read_csv("results/robogrammar/data/first_iteration.txt")
        evogym_experiment.plot_first_iteration(df, figpath, "RoboGrammar")

