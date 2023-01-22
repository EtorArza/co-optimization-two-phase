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
        from heuristic_search_algo_mpc import *

        sys.argv.pop()
        seed = 2
        mode = 'saveall'
        resfilepath = "../../results/robogrammar/data/first_iteration.txt"
        os.chdir("other_repos/RoboGrammar")


        torch.set_default_dtype(torch.float64)
        args_list = ['--grammar-file', 'data/designs/grammar_apr30.dot',
                    '--lr', '1e-4',
                    '--eps-start', '1.0',
                    '--eps-end', '0.1',
                    '--eps-decay', '0.3',
                    '--eps-schedule', 'exp-decay',
                    '--eps-sample-start', '1.0',
                    '--eps-sample-end', '0.1',
                    '--eps-sample-decay', '0.3',
                    '--eps-sample-schedule', 'exp-decay',
                    '--num-samples', '1', 
                    '--opt-iter', '25', 
                    '--batch-size', '32',
                    '--states-pool-capacity', '10000000',
                    '--depth', '10',
                    '--max-nodes', '20',
                    '--save-dir', './trained_models/',
                    '--log-interval', '100',
                    '--task', 'FlatTerrainTask',
                    '--eval-interval', '1000',
                    '--max-trials', '1000',
                    '--num-eval', '1',
                    '--num-iterations', '4',
                    '--mpc-num-processes', '4',
                    '--seed', str(seed),
                    '--no-noise',
                    ]

        solve_argv_conflict(args_list)
        parser = get_parser()
        args = parser.parse_args(args_list)

        if not args.test:
            args.save_dir = os.path.join(args.save_dir, args.task, get_time_stamp())
            try:
                os.makedirs(args.save_dir, exist_ok = True)
            except OSError:
                pass
            
            fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
            fp.write(str(args_list))
            fp.close()

        search_algo(args, resfilepath, mode)
        
    if sys.argv[1] == "--plot":
        import evogym_experiment
        import pandas as pd
        from matplotlib import pyplot as plt
        df = pd.read_csv("results/robogrammar/data/first_iteration.txt")
        evogym_experiment.plot_first_iteration(df, figpath)

