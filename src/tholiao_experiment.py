import sys
import os
import time

if sys.argv[1] == "--clean":
    os.system("rm other_repos/tholiao/logs/*.npy -f")
    os.system("rm V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/logs/models/20* -f")
    os.system("rm -f results/tholiao/data/*")
    os.system("rm -f results/tholiao/figures/*")
    os.system("rm -f results/tholiao/videos/*")



def local_launch_one(experiment_index):

    os.chdir("other_repos/tholiao")
    sys.path.append(sys.path[0]+"/../other_repos/tholiao")
    sys.path.append("/home/paran/Dropbox/BCAM/08_estancia_2/code/other_repos/tholiao")
    print(sys.path)
    from main import cli_main

    os.system("rm -f logs/params.npy")
    os.system("killall -9 vrep.sh")
    os.system("killall -9 vrep")
    time.sleep(5)
    # https://aweirdimagination.net/2020/06/28/kill-child-jobs-on-script-exit/
    os.system("cd ../../V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/ &&  ./vrep.sh -h &")
    time.sleep(5)


    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("tholiao", experiment_index)
    params.print_parameters()

    import random
    import numpy as np
    random.seed(params.seed)
    np.random.seed(params.seed)
 
    no = NestedOptimization("../../results/tholiao/data/", params)



    cli_main(no)


if sys.argv[1] == "--local_launch":
    experiment_index = int(sys.argv[2])
    sys.argv = sys.argv[:1]
    local_launch_one(experiment_index)


if sys.argv[1] == "--method_launch":
    experiment_index = int(sys.argv[2])
    sys.argv = sys.argv[:1]
    local_launch_one(experiment_index)


elif sys.argv[1] == "--sequential_launch":

    from itertools import product
    from NestedOptimization import convert_from_seconds, experimentProgressTracker
    import joblib

    progress_filename = "tholiao_sequential.txt"

    prog = experimentProgressTracker(progress_filename, 540, 555)
    
    def launch_next(prog: experimentProgressTracker):
        i = prog.get_next_index()

        exit_status = os.system(f"python src/tholiao_experiment.py --local_launch {i}")
        os.system("rm other_repos/tholiao/logs/*.npy -f")
        os.system("rm V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/logs/models/20* -f")
        if exit_status == 0:
            prog.mark_index_done(i)
        else:
            print(exit_status)
            exit(1)

    while not prog.done:
        launch_next(prog)
