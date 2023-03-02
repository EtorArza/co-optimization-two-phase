import sys
sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os
from NestedOptimization import NestedOptimization
from multiprocessing.managers import BaseManager
import itertools
from plot_src import *

figpath = "results/evogym/figures"

def get_sequence_of_parameters():
        import itertools
        seed_list = list(range(2,22))
        inners_per_outer_proportion_list = [1.0, 0.2] # Default is 1000
        inner_length_proportion_list = [1.0, 0.2] # Default is 64
        env_name_list = ["Walker-v0"]
        experiment_mode_list = ["reeval_each_vs_end"]
        max_frames_list = [4004000] # Default 32032000 considering 250 morphologies evaluated (easy tasks).

        res = list(itertools.product(seed_list, inners_per_outer_proportion_list, inner_length_proportion_list,env_name_list, experiment_mode_list, max_frames_list))
        # remove the combinations containining 2 different parameters != 1.0.
        res = [item for item in res if 1.0 in item or item[1] == item[2]]
        return res


def execute_experiment_locally(experiment_index):

    seed, inners_per_outer_proportion, inner_length_proportion, env_name, experiment_mode, max_frames = get_sequence_of_parameters()[experiment_index]

    random.seed(seed)
    np.random.seed(seed)
    os.chdir("other_repos/evogym/examples")

    # # Parallel
    # BaseManager.register('NestedOptimization', NestedOptimization)
    # manager = BaseManager()
    # manager.start()
    # no = manager.NestedOptimization(f"../../../results/evogym/data/{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}_{seed}.txt", mode, max_frames, inners_per_outer_proportion, inner_length_proportion)

    # Sequential

    no = NestedOptimization("../../../results/evogym/data/", experiment_mode, experiment_index, env_name, max_frames, inners_per_outer_proportion, inner_length_proportion, seed)

    run_ga(
        pop_size = 25,
        structure_shape = (5,5),
        no = no
    )
    os.chdir("../../..")



if __name__ == "__main__":
    if sys.argv[1] == "--local_launch":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --local_launch and i.\n\nUsage:\npython src/robogrammar_experiment.py i")
            exit(1)
        experiment_index = int(sys.argv[2])
        sys.argv = sys.argv[:1]
        seq_parameters = get_sequence_of_parameters()
        print("Total number of executions:", len(seq_parameters))
        print("Parameters current execution:",seq_parameters[experiment_index])
        # max_frames=32032000 is the default value if we consider 250 morphologies evaluated.
        execute_experiment_locally(experiment_index)



    elif sys.argv[1] == "--plot":
        print("Inner learning algorithm in evogym is ppo.")
        df = plot_comparison_parameters("results/evogym/data", figpath, "quantity")



    if sys.argv[1] == "--visualize":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --visualize and experiment_index.\n\nExample:\npython src/robogrammar_experiment.py --visualize 2")
            exit(1)
        experiment_index = int(sys.argv[2])
        sys.argv = sys.argv[:1]
        seq_parameters = get_sequence_of_parameters()
        print("Total number of executions:", len(seq_parameters))
        print("Parameters current execution:",seq_parameters[experiment_index])
        seed, inners_per_outer_proportion, inner_length_proportion, env_name, experiment_mode, max_frames = seq_parameters[experiment_index]

        random.seed(seed)
        np.random.seed(seed)
        os.chdir("other_repos/evogym/examples")

        from ga.run import load_visualization_data, save_robot_gif_standalone

        for best_or_current in ["current","best"]:
            pickle_dump_path = f"simulation_objects_{experiment_index}_{best_or_current}.pkl"
            out_path, env_name, structure, ctrl_path = load_visualization_data(pickle_dump_path)
            save_robot_gif_standalone(out_path, env_name, structure, ctrl_path)

    else:
        ValueError("sys.argv[1] was ", sys.argv[1], " and this is not a recognized experiment.")
  