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


def cli_main(seed, max_frames, inners_per_outer_proportion, inner_length_proportion, experiment_index):
    obj_f=0
    optimizer='hpcbbo'
    init_uc = 5 # Number of initial control optimization loops
    init_cn = 5 # Number of initial morphology optimization loops
    uc_runs_per_cn = 50 # Ratio of morphology opt. loops to opt. control loops
    batch_size = 5 # Morphology batch size
    total = 100 # Number of total morphology optimization loops
    obj_f = 0 # Switch between different objective functions
    contextual = "store_true" # Toggle contextual optimization. Only meaningful for optimizers 'hpcbbo' and 'bayesopt'
    popsize = -1 # Only relevant for CMA-ES
    num_inputs = N_CTRL_PARAMS[obj_f] + N_MRPH_PARAMS[obj_f]
    joint_bounds = np.hstack((np.array(CONTROLLER_BOUNDS[obj_f]),
                              np.array(MORPHOLOGY_BOUNDS[obj_f])))
    no = NestedOptimization(f"../../../results/evogym/data/flatterrain_{max_frames}_{inners_per_outer_proportion}_{inner_length_proportion}_{seed}.txt", "standard", max_frames, inners_per_outer_proportion, inner_length_proportion, experiment_index)

    if obj_f == 0:
        sim = HwSwDistSim()
    elif obj_f == 1:
        sim = JointObj2()
    elif obj_f == 2:
        sim = JointObj3()
    else:
        raise ValueError("Objective function must be: 0, 1, or 2")

    if optimizer == 'bayesopt':
        optimizer = BayesOptimizer(obj_f=sim.get_obj_f(max_steps=401),
                                   num_inputs=num_inputs,
                                   bounds=joint_bounds,
                                   n_init=init_uc)
    elif optimizer == 'cmaes':
        optimizer = CMAESOPtimizer(obj_f=sim.get_obj_f(max_steps=401),
                                   bounds=joint_bounds,
                                   popsize=popsize)
    elif optimizer == 'random':
        optimizer = RandomOptimizer(obj_f=sim.get_obj_f(max_steps=401),
                                   num_inputs=num_inputs,
                                   bounds=joint_bounds,
                                   n_init=init_uc)
    elif optimizer == 'hpcbbo':
        optimizer = JointBatchOptimizer(obj_f=sim.get_obj_f(max_steps=401),
                                  n_uc=N_CTRL_PARAMS[obj_f],
                                  init_uc=init_uc,
                                  bounds_uc=CONTROLLER_BOUNDS[obj_f],
                                  uc_runs_per_cn=uc_runs_per_cn,
                                  init_cn=init_cn,
                                  bounds_cn=MORPHOLOGY_BOUNDS[obj_f],
                                  n_cn=N_MRPH_PARAMS[obj_f],
                                  batch_size=batch_size,
                                  contextual=contextual)

    optimizer.optimize(no, total=total)
    sim.exit()

