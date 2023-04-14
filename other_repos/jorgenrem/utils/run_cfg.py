

"""
Command line utility to run one or more CFG files
"""

from concurrent.futures import ThreadPoolExecutor
from termcolor import colored, cprint
import argparse
import functools
import itertools
import os.path
import subprocess
import sys
import tqdm
# Path to where 'run.py' is located
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_cfg(cfg_seed, output, threads, prog=None):
    """Helper function to run a single configuration"""
    cfg_initial, seed = cfg_seed
    cfg, initial = cfg_initial
    # Create output path based on the given folder 'output'
    cfg_name = os.path.basename(os.path.splitext(cfg)[0])
    output = os.path.join(output, cfg_name)
    exp = ['./run.py',
           '--file', cfg,
           '--shared', '',
           '--output', str(output),
           '--seed', str(seed),
           '--parallel', 'thread',
           '--processes', str(threads)]
    if initial:
        exp.extend(['--population', initial])
    try:
        subprocess.run(exp, check=True, cwd=RUN_DIR)
    except subprocess.CalledProcessError:
        msg = "Configuration '{!s}' failed!".format(cfg)
        if not prog:
            cprint(msg, 'red', file=sys.stderr)
        else:
            prog.write(colored(msg, 'red'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=argparse.FileType('r'),
                        required=True,
                        help="File with seeds to simulate")
    parser.add_argument('--output', default='.',
                        help="Output location for results")
    parser.add_argument('cfg_files', nargs='+',
                        help="Configuration files to run")
    parser.add_argument('--processes', type=int, default=1,
                        help="Number of parallel processes to run")
    parser.add_argument('--threads', type=int, default=1,
                        help="Number of threads per process")
    parser.add_argument('--initial', nargs='+', default=[],
                        help="Initial population(s) to start from")
    args = parser.parse_args()
    # Create reproducible seeds for repeated runs
    reps = []
    for line in args.seeds:
        if not (line.startswith('#') or line.startswith('//')):
            reps.append(int(line.strip()))
    # Create progress bar for status messages etc.
    prog = tqdm.tqdm(total=len(reps) * len(args.cfg_files),
                     desc="Preparing run",
                     unit="run")
    # Ensure that we have the absolute path to the configuration files
    files = [os.path.abspath(fil) for fil in args.cfg_files]
    # Check number of initial populations given
    if args.initial:
        initial = [os.path.abspath(i) for i in args.initial]
        if len(initial) == len(files):
            files = zip(files, initial)
        elif len(initial) == 1:
            files = zip(files, itertools.repeat(initial[0]))
        else:
            sys.exit(colored("Wrong number of initial populations given, need either same number as 'cfg_files' or 1", 'red'))
    else:
        files = zip(files, itertools.repeat(None))
    # Ensure result path exist
    output = os.path.abspath(args.output)
    if not os.path.exists(output):
        prog.write(colored("Creating output folder '{!s}'".format(args.output),
                           'yellow'))
        os.makedirs(output)
    # Create curried function for 'map' below
    exe_func = functools.partial(_run_cfg, output=output, threads=args.threads,
                                 prog=prog)
    # Run configurations
    prog.set_description("Running configurations")
    with ThreadPoolExecutor(max_workers=args.processes) as exe:
        cfgs = itertools.product(files, reps)
        for _ in exe.map(exe_func, cfgs):
            prog.update()
