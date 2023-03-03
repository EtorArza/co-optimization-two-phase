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

def get_sequence_of_parameters():
    seed_list = list(range(2,22))
    inner_quantity_proportion_list = [1.0, 0.5] # Default is 1000
    inner_length_proportion_list = [1.0, 0.5] # Default is 64
    return list(itertools.product(seed_list, inner_quantity_proportion_list, inner_length_proportion_list))


if sys.argv[1] == "--local_launch":
    os.system("rm -f logs/params.npy")
    os.system("killall -9 vrep.sh")
    os.system("killall -9 vrep")
    time.sleep(5)
    os.system("cd ../../V-REP_PRO_EDU_V3_6_2_Ubuntu18_04/ &&  ./vrep.sh -h &")
    time.sleep(5)

    experiment_index = int(sys.argv[2])
    sys.argv = sys.argv[:1]
    seq_parameters = get_sequence_of_parameters()
    print("Total number of executions:", len(seq_parameters))
    print("Parameters current execution:",seq_parameters[experiment_index])
    seed, inner_quantity_proportion, inner_length_proportion = seq_parameters[experiment_index]
    # max_frames=9999999999 is the default value.
    max_frames = 99999999999
    cli_main(seed, max_frames, inner_quantity_proportion, inner_length_proportion, experiment_index)
