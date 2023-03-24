import gym_rem2D.envs.Modular2DEnv
import REM2D_main as r2d
import datetime
import os
import argparse
import numpy as np
import time


class stopwatch:

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t


STOPWATCH = stopwatch()
r2d.STOPWATCH = STOPWATCH

parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('--method', required=True, metavar='method', type=str, help='Must be constant, bestasref, or bestevery.', default=None, nargs='?')
parser.add_argument('--seed', required=True, metavar='seed', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--gracetime', required=True, metavar='gracetime', type=int, help='Grace time parameter', default=None, nargs='?')
parser.add_argument('--res_filepath', required=True, metavar='res_filepath', type=str, help='Result file path', default=None, nargs='?')

args = parser.parse_args()


def run_experiment(exp_dir):
	config, dir = r2d.setup(directory=exp_dir)	
	experiment = r2d.run2D(config,dir)
	experiment.run(config)


