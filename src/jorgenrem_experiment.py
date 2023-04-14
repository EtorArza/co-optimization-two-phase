import sys
import os



def launch_one(experiment_index):
    from NestedOptimization import Parameters, NestedOptimization
    previousDir = os.getcwd()
    os.chdir("other_repos/jorgenrem")
    params = Parameters("jorgenrem", experiment_index)
    no = NestedOptimization("../../results/jorgenrem/data", params)
    params.print_parameters()
    sys.path.append(sys.path[0]+"/../other_repos/jorgenrem/")
    from run import main
    main(no)
    os.chdir(previousDir)

launch_one(6)
