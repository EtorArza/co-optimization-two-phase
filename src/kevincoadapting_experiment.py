import sys
import os
from plot_src import *
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # Adding parent directory to sys.path
from other_repos.kevincoadapting.main import main  # Importing main.py
import other_repos.kevincoadapting.experiment_configs as cfg
from NestedOptimization import NestedOptimization, Parameters



def launch_one(experiment_index):
    
    config = cfg.config_dict["sac_pso_batch"]
    params = Parameters("kevincoadapting", experiment_index)
    no = NestedOptimization("results/kevincoadapting/data", params)
    params.print_parameters()
    main(config, no)

if len(sys.argv) > 3 or len(sys.argv) <= 1:
    print("Usage: python kevincoadapting.py [--clean | --local_launch | --visualize]")
    exit(1)

elif sys.argv[1] == "--clean":
    os.system("rm -f results/kevincoadapting/data/*")
    os.system("rm -f results/kevincoadapting/figures/*")
    os.system("rm -f results/kevincoadapting/videos/*")
    os.system("rm -f other_repos/kevincoadapting/data_exp_sac_pso_batch/* -rf")
    os.system("rm -f data_exp_sac_pso_batch/* -rf")
    print("Cleaned experiment files.")
    exit(0)


elif sys.argv[1] == "--local_launch":
    launch_one(int(sys.argv[2]))

elif sys.argv[1] == "--visualize":
    pass
    # params = Parameters("jorgenrem", int(sys.argv[2]))
    # no = NestedOptimization("../../results/jorgenrem/data", params)
    # modular_er.eval.animate_from_dump(f"other_repos/jorgenrem/dumps_for_animation/animation_dump_current{int(sys.argv[2])}.wb")
    # modular_er.eval.animate_from_dump(f"other_repos/jorgenrem/dumps_for_animation/animation_dump_best{int(sys.argv[2])}.wb")

else:
    raise ValueError(f"Argument {sys.argv[1]} not recognized.")
