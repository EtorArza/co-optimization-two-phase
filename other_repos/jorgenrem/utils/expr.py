#!/usr/bin/env python

"""
Command line utility to run several experiments with different configurations
"""
from termcolor import colored
import argparse
import configparser
import io
import itertools
import os.path
import subprocess
import tqdm

# Path for experiment results
result_path = "rdlf_res/param_sweep/"
# Path to where 'run.py' is located
run_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The different EAs to run
eas = ['single']
# Different crossover rates to test
# Controller mutation rate
morph_mutation = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
# Controller perturbation size
ctrl_sigma = [0.005, 0.01, 0.05, 0.1, 0.2]
# Create combination of all parameters to test
parameters = itertools.product(eas, morph_mutation, ctrl_sigma)
# Create progress bar for better indication of progress
prog = tqdm.tqdm(list(parameters),
                 desc="Parameter sweep", unit='configuration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', action='store_true',
                        help="Store configuration instead of running")
    args = parser.parse_args()
    for i, config in enumerate(prog):
        # Create storage name for configuration
        out_str = '_'.join(map(str, config)).replace('.', '-')
        out_name = os.path.abspath(os.path.join(result_path, out_str + '.zip'))
        # Split out parameters to run
        ea, morph, sigma = config
        # Create configuration to pass to the process
        cfg = configparser.ConfigParser()
        cfg['experiment'] = {'ea': ea,
                             'checkpoint_frequency': '50'}
        # Setup EA specific configuration
        if ea == 'map-elites':
            cfg['ea'] = {'initial_size': '1000',
                         'evaluations': '100000',
                         'selection': 'tournament',
                         'tournament_size': '5',
                         'batch_size': '200'}
        elif ea == 'single':
            cfg['ea'] = {'selection': 'tournament',
                         'tournament_size': '2',
                         'population': '200',
                         'elitism': 10,
                         'evaluations': '100000'}
        elif ea == 'nsga2':
            cfg['ea'] = {'population': '200',
                         'evaluations': '100000',
                         'distance': 'distance',
                         'penalty_func': 'exp'}
        # Setup shared configuration
        cfg['ea']['crossover_prob'] = '0.2'
        cfg['ea']['mutation_prob'] = '1.0'
        cfg['morphology'] = {}
        cfg['morphology']['max_size'] = '20'
        cfg['morphology']['max_depth'] = '4'
        cfg['evaluation'] = {}
        cfg['evaluation']['time'] = '20.0'
        cfg['evaluation']['warm_up'] = '2.0'
        cfg['evaluation']['environment'] = 'ModularLocomotion3D-v0'
        cfg['encoding'] = {}
        cfg['encoding']['type'] = 'direct'
        cfg['encoding']['morphology_prob'] = str(morph)
        cfg['control'] = {}
        cfg['control']['sigma'] = str(sigma)
        cfg['control']['type'] = 'wave'
        if args.store:
            file_path = os.path.join("param_sweep/parameters",
                                     "{!s}.cfg".format(i))
            with open(file_path, 'w') as fil:
                cfg.write(fil)
        else:
            # Convert configuration to string that we can pass in on the
            # command line
            cfg_str = io.StringIO(newline='')
            cfg.write(cfg_str)
            # Run experiment as subprocess
            exp = ['./run.py', '--file', '-', '--shared', '', '--quiet',
                   '--seed', '12345',
                   '--output', out_name,
                   '--parallel', 'thread',
                   '--processes', '32',
                   '--chunksize', '1']
            try:
                subprocess.run(exp, input=cfg_str.getvalue(), check=True,
                               encoding='utf-8', cwd=run_dir)
            except subprocess.CalledProcessError as e:
                prog.write(colored("Experiment: '{!s}' failed!"
                                   .format(out_str),
                                   'red'))
                prog.write(colored("Traceback:\n{!s}".format(e), 'yellow'))
