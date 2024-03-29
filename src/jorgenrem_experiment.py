import sys
import os
from plot_src import *


if sys.argv[1] == "--clean":
    os.system("rm -f results/jorgenrem/data/meth*")
    os.system("rm -f results/jorgenrem/figures/meth*")
    os.system("rm -f results/jorgenrem/videos/meth*")
    os.system("rm -f other_repos/jorgenrem/dumps_for_animation/anim*")
    print("Cleaned experiment files.")
    exit(0)





def launch_one(experiment_index):
    from NestedOptimization import Parameters, NestedOptimization
    previousDir = os.getcwd()
    os.chdir("other_repos/jorgenrem")
    params = Parameters("jorgenrem", experiment_index)
    no = NestedOptimization("../../results/jorgenrem/data", params)
    params.print_parameters()
    print("os.getcwd() = ", os.getcwd())
    sys.path.append(sys.path[0]+"/../other_repos/jorgenrem/")
    sys.path.append(os.getcwd())
    from run import main
    sys.argv = [sys.argv[0]]
    main(no)
    os.chdir(previousDir)


def launch_one_parameter_tuning(i):
    from itertools import product
    from NestedOptimization import Parameters, NestedOptimization

    seeds_tune = list(range(20))
    tuning_parameter_combs = list(product(seeds_tune, [1, 8, 16, 32, 128, 512]))

    seed, default_inner_quantity = tuning_parameter_combs[i]
    print("seed, default_inner_quantity = ", seed, default_inner_quantity)

    previousDir = os.getcwd()
    os.chdir("other_repos/jorgenrem")
    params = Parameters("jorgenrem", 1)
    params.seed = seed
    def return_filename():
        return f"paramtuning_{default_inner_quantity}_{seed}"
    params.get_result_file_name=return_filename
    params._default_inner_quantity = default_inner_quantity
    params._inner_quantity_proportion = 1.0 
    params._inner_length_proportion = 1.0

    params.print_parameters()
    no = NestedOptimization("../../results/jorgenrem/data", params, deletePreviousResults=False, limit_the_amount_of_written_lines=True if default_inner_quantity in [1,8] else False)
    sys.path.append(sys.path[0]+"/../other_repos/jorgenrem/")
    print(sys.path)
    from run import main
    import modular_er.eval
    modular_er.eval.save_data_animation = lambda dump_path, video_label, individual, controller, no, seconds, max_size, env: None
    sys.argv = [sys.argv[0]]
    main(no)
    os.chdir(previousDir)



if sys.argv[1] == "--local_launch":
    import itertools
    import time
    launch_one(int(sys.argv[2]))

elif sys.argv[1] == "--local_launch_tune":
    import itertools
    import time
    assert len(sys.argv) == 3
    launch_one_parameter_tuning(int(sys.argv[2]))

elif sys.argv[1] == "--visualize":
    from NestedOptimization import Parameters, NestedOptimization

    print("Generating a figure completely wrecks pybullet, where the AABB calculation is different.")    
    print("This makes it impossible to generate the animations in my laptop and reproduce the results during the training.")    

    print("see https://github.com/bulletphysics/bullet3/issues/4502")    
    # import numpy as np
    # from matplotlib import pyplot as plt
    # plt.figure()
    # image_data = np.random.random((100, 100))
    # plt.imshow(image_data, cmap='gray')
    # plt.close()


    params = Parameters("jorgenrem", int(sys.argv[2]))
    params.print_parameters()

    no = NestedOptimization("../../results/jorgenrem/data", params)
    sys.path.append(sys.path[0]+"/../other_repos/jorgenrem/")
    print(sys.path)
    import modular_er.eval 

    from matplotlib import pyplot as plt
    print("saving animation...", end="")
    modular_er.eval.animate_from_dump(f"other_repos/jorgenrem/dumps_for_animation/animation_dump_current{int(sys.argv[2])}.wb")
    modular_er.eval.animate_from_dump(f"other_repos/jorgenrem/dumps_for_animation/animation_dump_best{int(sys.argv[2])}.wb")

elif sys.argv[1] == "--local_launch_tune_sequentially":
    from NestedOptimization import convert_from_seconds, experimentProgressTracker
    import joblib
    import pandas as pd
    import time

    progress_filename = "jorgenremtune_progress_report.txt"
    start_index = 0
    end_index = 100
    prog = experimentProgressTracker(progress_filename, start_index, end_index)
    while not prog.done:
        i = prog.get_next_index()
        exit_status = os.system(f"python src/jorgenrem_experiment.py --local_launch_tuning {i}")
        if exit_status == 0:
            prog.mark_index_done(i)
        else:
            print(exit_status)
            exit(1)



elif sys.argv[1] == "--plot_tune":
    plot_tune("results/jorgenrem/data", "results/jorgenrem/figures")

elif sys.argv[1] == "--plot":
    df = plot_comparison_parameters("gym-rem", "results/jorgenrem/data", "results/jorgenrem/figures")

else:
    raise ValueError(f"Argument {sys.argv[1]} not recognized.")