import sys
import os

if sys.argv[1] == "--clean":    
    os.system("rm other_repos/evogym/examples/controller_to_generate_animation_* -fv")
    os.system("rm other_repos/evogym/examples/simulation_objects_* -fv")
    exit(0)
elif sys.argv[1] == "--cleanall":    
    os.system("rm other_repos/evogym/examples/controller_to_generate_animation_* -fv")
    os.system("rm other_repos/evogym/examples/simulation_objects_* -fv")
    os.system("rm results/evogym/data/*.txt -fv")
    os.system("rm results/evogym/videos/*.gif -fv")
    os.system("rm results/evogym/figures/*.pdf -fv")
    exit(0)

elif sys.argv[1] == "--cleanfigs":    
    os.system("rm results/evogym/figures/*.pdf -fv")
    exit(0)




sys.path.append("./other_repos/evogym/examples") 
import random
import numpy as np
from ga.run import run_ga
import os
from NestedOptimization import NestedOptimization, Parameters
from multiprocessing.managers import BaseManager
import itertools
from plot_src import *

figpath = "results/evogym/figures"



def execute_experiment_locally(experiment_index):

    params = Parameters("evogym", experiment_index)
    params.print_parameters()


    random.seed(params.seed)
    np.random.seed(params.seed)
    os.chdir("other_repos/evogym/examples")

    # # Parallel
    # BaseManager.register('NestedOptimization', NestedOptimization)
    # manager = BaseManager()
    # manager.start()
    # no = manager.NestedOptimization(f"../../../results/evogym/data/{max_frames}_{inner_quantity_proportion}_{inner_length_proportion}_{seed}.txt", mode, max_frames, inner_quantity_proportion, inner_length_proportion)

    # Sequential
    no = NestedOptimization("../../../results/evogym/data", params)
    run_ga(pop_size = 25, structure_shape = (5,5), no = no)



if __name__ == "__main__":
    if sys.argv[1] == "--local_launch":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --local_launch and i.\n\nUsage:\npython src/robogrammar_experiment.py i")
            exit(1)
        experiment_index = int(sys.argv[2])
        sys.argv = sys.argv[:1]
        execute_experiment_locally(experiment_index)



    elif sys.argv[1] == "--plot":
        print("Inner learning algorithm in evogym is ppo.")
        df = plot_comparison_parameters("results/evogym/data", figpath)

    elif sys.argv[1] == "--reindex_result_files":
        print("Reindexing results files...")
        params = Parameters("evogym", 0)
        params.reindex_all_result_files("results/evogym/data", ".txt")
        params.reindex_all_result_files("results/evogym/videos", ".gif")


    elif sys.argv[1] == "--visualize":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --visualize and experiment_index.\n\nExample:\npython src/robogrammar_experiment.py --visualize 2")
            exit(1)
        experiment_index = int(sys.argv[2])
        sys.argv = sys.argv[:1]
        params = Parameters("evogym", experiment_index)


        random.seed(params.seed)
        np.random.seed(params.seed)
        os.chdir("other_repos/evogym/examples")

        from ga.run import load_visualization_data, save_robot_gif_standalone

        for best_or_current in ["current","best"]:
            pickle_dump_path = f"simulation_objects_{experiment_index}_{best_or_current}.pkl"
            out_path, env_name, structure, ctrl_path = load_visualization_data(pickle_dump_path)
            save_robot_gif_standalone(out_path, env_name, structure, ctrl_path)

    elif sys.argv[1] == "--cluster_launch":
        print("Launching evogym in cluster...")
        params = Parameters("evogym", 0)
        n = len(params.get_n_experiments())
        print(f"Run the following command: \nsbatch --array=0-{n-1} cluster_scripts/launch_one_evogym.sl\n")




    else:
        raise ValueError("sys.argv[1] was ", sys.argv[1], " and this is not a recognized experiment.")
  