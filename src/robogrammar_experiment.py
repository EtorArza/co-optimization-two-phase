import sys
import itertools
figpath = "results/robogrammar/figures"



def get_sequence_of_parameters():
    seed_list = list(range(2,22))
    inners_per_outer_proportion_list = [1.0, 0.5, 0.25] # Default is 64
    inner_length_proportion_list = [1.0, 0.5, 0.25] # Default is 128
    res = list(itertools.product(seed_list, inners_per_outer_proportion_list, inner_length_proportion_list))
    res = [item for item in res if not (0.25 in item and 0.5 in item)]
    return res

def execute_experiment_locally(seed, max_frames, inners_per_outer_proportion, inner_length_proportion, experiment_index):
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
    cpus = 8
    env_name = 'FlatTerrainTask'
    experiment_name = f"{experiment_index}_{env_name}_{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}_{seed}"

    resfilepath = f"../../results/robogrammar/data/{experiment_name}.txt"


    torch.set_default_dtype(torch.float64)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'graph_learning'))
    sys.path.append(os.path.join(base_dir, 'design_search'))
    from NestedOptimization import NestedOptimization
    import os
    no = NestedOptimization(resfilepath, mode, max_frames, inners_per_outer_proportion, inner_length_proportion, experiment_index, experiment_name)
    main(no, algorithm, cpus, env_name, seed)


if __name__ == "__main__":


    if sys.argv[1] == "--local_launch":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --local_launch and i.\n\nUsage:\npython src/robogrammar_experiment.py i")
            exit(1)
        experiment_index = int(sys.argv[2])
        seq_parameters = get_sequence_of_parameters()
        print("Total number of executions:", len(seq_parameters))
        print("Parameters current execution:",seq_parameters[experiment_index])
        seed, inners_per_outer_proportion, inner_length_proportion = seq_parameters[experiment_index]
        # max_frames=40960000 is the default value if we consider 5000 iterations as in the example in the github.
        execute_experiment_locally(seed, 40960000, inners_per_outer_proportion, inner_length_proportion, experiment_index)

        
    elif sys.argv[1] == "--plot":
        from plot_src import *
        import pandas as pd
        from matplotlib import pyplot as plt
        print("Inner learning algorithm in evogym is MPC.")


    elif sys.argv[1] == "--visualize":
        from viewer import generate_video, unpickle_data_for_video_generation
        if sys.executable.split('/')[-3] != 'venv':
            print("This script requires that conda is deactivated and the python environment in other_repos/RoboGrammar/venv/bin/activate is activated. To achieve this, run the following: \n\nconda deactivate\nsource other_repos/RoboGrammar/venv/bin/activate")
            print("\n\nOnce 'venv' has been loaded, rerun this script.")
            exit(1)

        import torch
        import os
        experiment_index = sys.argv[2]
        os.chdir("other_repos/RoboGrammar")
        # sys.path.append("./other_repos/RoboGrammar/examples/graph_learning")
        # from heuristic_search_algo_mpc import *
        sys.path.append("./other_repos/RoboGrammar/examples/design_search")
        from design_search import main

        sys.argv.pop()
        sys.argv.pop()

        for mode in ["current","best"]:
        
            task, robot, opt_seed, input_sequence, visualization_path = unpickle_data_for_video_generation(experiment_index, mode)
            save_obj_dir = f"tmp_{experiment_index}"

            generate_video(task, robot, opt_seed, input_sequence, save_obj_dir, visualization_path)


    # elif sys.argv[1] == "--cluster_launch":
        