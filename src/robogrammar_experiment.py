import sys
import os
figpath = "results/robogrammar/figures"

if sys.argv[1] == "--clean":    
    os.system("rm other_repos/RoboGrammar/rule_sequence_* -f")
    os.system("rm other_repos/RoboGrammar/simulation_objects_* -f")
    exit(0)
elif sys.argv[1] == "--cleanall":    
    os.system("rm other_repos/RoboGrammar/rule_sequence_* -f")
    os.system("rm other_repos/RoboGrammar/simulation_objects_* -f")
    os.system("rm results/robogrammar/data/proposedmethod*.txt -f")
    os.system("rm results/robogrammar/videos/proposedmethod*.mp4 -f")
    os.system("rm results/robogrammar/figures/proposedmethod*.pdf -f")
    exit(0)

elif sys.argv[1] == "--cleanfigs":    
    os.system("rm results/robogrammar/figures/*.pdf -fv")
    exit(0)



def execute_experiment_locally(experiment_index):



    if sys.executable.split('/')[-3] != 'venv':
        print("This script requires that conda is deactivated and the python environment in other_repos/RoboGrammar/venv/bin/activate is activated. To achieve this, run the following: \n\nconda deactivate\nsource other_repos/RoboGrammar/venv/bin/activate")
        print("\n\nOnce 'venv' has been loaded, rerun this script.")
        exit(1)

    import torch
    import os
    sys.path.insert(0,os.path.abspath('other_repos/RoboGrammar/build/examples/python_bindings'))

    os.chdir("other_repos/RoboGrammar")
    # sys.path.append("./other_repos/RoboGrammar/examples/graph_learning")
    # from heuristic_search_algo_mpc import *
    sys.path.append("./other_repos/RoboGrammar/examples/design_search")
    from design_search import main

    sys.argv.pop()
    sys.argv.pop()
    algorithm = ["mcts", "random"][0]
    cpus = 32

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
        df = plot_comparison_parameters("results/robogrammar/data", figpath)
        df = plot_proposedmethod("results/evogym/data", figpath)

    elif sys.argv[1] == "--reindex_result_files":
        print("Reindexing results files...")
        from NestedOptimization import Parameters

        params = Parameters("robogrammar", 0)
        params.reindex_all_result_files("results/robogrammar/videos", ".mp4")
        params.reindex_all_result_files("results/robogrammar/data", ".txt")


    elif sys.argv[1] == "--visualize":
        import os
        sys.path.insert(0,os.path.abspath('other_repos/RoboGrammar/build/examples/python_bindings'))

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


    elif sys.argv[1] == "--move_anomaly_videos":
        import subprocess
        # standard
        for i, method in zip([8, 17, 4, 13, 0, 9], ["standard"]*2+["reduced_length"]*2 + ["reduced_quantity"]*2):
            result = subprocess.run(f"ls results/robogrammar/videos/reevaleachvsend_{i}_*current.mp4", shell=True, capture_output=True, text=True)
            file_names = result.stdout.strip().split('\n')
            assert len(file_names) == 1
            file_name = file_names[0]
            os.system(f"ffmpeg -i {file_name} -vf \"scale=320:240\" -r 10 animations_for_the_paper/tiny_quantity_vs_length_robogrammar/gif_{method}_{i}_robogrammar.gif")


    elif sys.argv[1] == "--cluster_launch":
        print("Command to launch evogym in cluster...")
        from NestedOptimization import Parameters
        params = Parameters("robogrammar", 0)
        print(f"sbatch --array=0-{params.get_n_experiments()-1} cluster_scripts/launch_one_robogrammar.sl")

    # elif sys.argv[1] == "--cluster_launch":
        