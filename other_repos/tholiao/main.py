# !/usr/bin/env python -W ignore::DeprecationWarning

import argparse
import time
import sys
from NestedOptimization import NestedOptimization

import numpy as np

from objectives import HwSwDistSim, JointObj2, JointObj3
from optimizers import JointBatchOptimizer, \
    CMAESOPtimizer, \
    RandomOptimizer, \
    BayesOptimizer
from utils import *


def cli_main(no):
    obj_f=0
    optimizer='hpcbbo'
    init_uc = 2 # Number of initial control optimization loops (To initialize the gaussian process)
    init_cn = 2 # Number of initial morphology optimization loops (To initialize the gaussian process)
    uc_runs_per_cn = 6 # Ratio of morphology opt. loops to opt. control loops
    batch_size = 2 # Morphology batch size
    total = 100 # Number of total morphology optimization loops
    obj_f = 0 # Switch between different objective functions
    contextual = False # Toggle contextual optimization. Contextual=False means that a new GP is initialized for each morphology evaluation.
    popsize = -1 # Only relevant for CMA-ES
    num_inputs = N_CTRL_PARAMS[obj_f] + N_MRPH_PARAMS[obj_f]
    joint_bounds = np.hstack((np.array(CONTROLLER_BOUNDS[obj_f]),
                              np.array(MORPHOLOGY_BOUNDS[obj_f])))

    if obj_f == 0:
        sim = HwSwDistSim()
    elif obj_f == 1:
        sim = JointObj2()
    elif obj_f == 2:
        sim = JointObj3()
    else:
        raise ValueError("Objective function must be: 0, 1, or 2")

    optimizer = JointBatchOptimizer(obj_f=sim.get_obj_f(no=no),
                                  n_uc=N_CTRL_PARAMS[obj_f],
                                  init_uc=init_uc,
                                  bounds_uc=CONTROLLER_BOUNDS[obj_f],
                                  uc_runs_per_cn=uc_runs_per_cn,
                                  init_cn=init_cn,
                                  bounds_cn=MORPHOLOGY_BOUNDS[obj_f],
                                  n_cn=N_MRPH_PARAMS[obj_f],
                                  batch_size=batch_size,
                                  contextual=contextual,
                                  no=no)
    optimizer.optimize(total=total)
    sim.exit()

