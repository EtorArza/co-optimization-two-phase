import sys
figpath = "results/robogrammar/figures"



def execute_experiment_locally(experiment_index):
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
    algorithm = ["mcts", "random"][0]
    cpus = 8

    resfilepath = f"../../results/robogrammar/data"


    torch.set_default_dtype(torch.float64)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'graph_learning'))
    sys.path.append(os.path.join(base_dir, 'design_search'))
    from NestedOptimization import NestedOptimization, Parameters
    import os
    params = Parameters("robogrammar", experiment_index)
    params.print_parameters()
    no = NestedOptimization(resfilepath, params)
    main(no, algorithm, cpus)


if __name__ == "__main__":


    if sys.argv[1] == "--local_launch":
        if len(sys.argv) != 3:
            print("ERROR: 2 parameters are required, --local_launch and i.\n\nUsage:\npython src/robogrammar_experiment.py i")
            exit(1)
        experiment_index = int(sys.argv[2])
        # max_frames=40960000 is the default value if we consider 5000 iterations as in the example in the github.
        execute_experiment_locally(experiment_index)

        
    elif sys.argv[1] == "--plot":
        from plot_src import *
        print("Inner learning algorithm in robogrammar is MPC.")
        df = plot_comparison_parameters("results/robogrammar/data", figpath, "length")


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


    elif sys.argv[1] == "--cluster_launch":
        print("Launching evogym in cluster...")
        from NestedOptimization import Parameters
        params = Parameters("robogrammar", 0)
        print(f"sbatch --array=0-{params.get_n_experiments()-1} cluster_scripts/launch_one_robogrammar.sl")

    # elif sys.argv[1] == "--cluster_launch":
        