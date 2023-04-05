from argparse import ArgumentError

import subprocess
import time
import re
from os.path import exists
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
import argparse



def launch_one(experiment_index):
    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("gymrem2d", experiment_index)
    params.print_parameters()

    no = NestedOptimization("../../../results/gymrem2d/data", params)
    sys.path.append(sys.path[0]+"/../other_repos/gymrem2d/ModularER_2D/")
    print(sys.path)
    import REM2D_main
    from REM2D_main import setup, run2D
    config, dir = setup(no)
    experiment = run2D(no,config,dir)
    experiment.run(config)



if sys.argv[1] == "--local_launch":
    import itertools
    import time
    launch_one(int(sys.argv[2]))


elif sys.argv[1] == "--visualize":
    from NestedOptimization import Parameters, NestedOptimization

    params = Parameters("gymrem2d", int(sys.argv[2]))
    params.print_parameters()

    no = NestedOptimization("../../../results/gymrem2d/data", params)
    sys.path.append(sys.path[0]+"/../other_repos/gymrem2d/ModularER_2D/")
    print(sys.path)
    import REM2D_main
    from REM2D_main import animate_from_dump

    animate_from_dump(f"other_repos/gymrem2d/dumps_for_animation/animation_dump_current{int(sys.argv[2])}.wb")
    animate_from_dump(f"other_repos/gymrem2d/dumps_for_animation/animation_dump_best{int(sys.argv[2])}.wb")
    