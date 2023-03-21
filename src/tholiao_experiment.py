# https://aweirdimagination.net/2020/06/28/kill-child-jobs-on-script-exit/
import sys
import os
import time
import itertools
os.chdir("other_repos/tholiao")
sys.path.append(sys.path[0]+"/../other_repos/tholiao")
sys.path.append("/home/paran/Dropbox/BCAM/08_estancia_2/code/other_repos/tholiao")
print(sys.path)

from main import cli_main



if sys.argv[1] == "--local_launch":
    os.system("rm -f logs/params.npy")
    os.system("killall -9 vrep.sh")
    os.system("killall -9 vrep")
    time.sleep(5)
    os.system("cd ../../V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/ &&  ./vrep.sh -h &")
    time.sleep(5)

    experiment_index = int(sys.argv[2])
    sys.argv = sys.argv[:1]

    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("tholiao", experiment_index)
    params.print_parameters()

    import random
    import numpy as np
    random.seed(params.seed)
    np.random.seed(params.seed)
 
    no = NestedOptimization("../../results/tholiao/data/", params)



    cli_main(no)
