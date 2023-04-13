#!/usr/bin/env python
"""
Command line utility to extract individuals from stored data and simulate
"""
from matplotlib.widgets import CheckButtons
from modular_er.ea.map_elites import Map, size_behavior
from modular_er.eval import evaluate
from termcolor import cprint, colored
import argparse
import gym
import gym_rem
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import pybullet as pyb
import time
import zipfile


def _evaluate(indiv, seconds=10.0, max_size=None, view=False, record=None, env="flat"):
    """Helper function to simulate individual"""
    # Setup environment
    env = gym.make(env)
    env.render()
    obs = env.reset(morphology=indiv.morphology, max_size=max_size)
    time.sleep(1)
    # Setup controllers
    ctrls = [m.ctrl for m in env.morphology if m.joint is not None]
    for m in env.morphology:
        if m.joint:
            m.ctrl.reset(m)
    steps = int(seconds / env.dt)
    if not view:
        input(colored("Press any key to start evaluation", 'green'))
        if record:
            log_id = env.client.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4,
                                                  fileName=record)
            time.sleep(1)
        for i in range(steps):
            obs = zip(*np.split(obs, 3))
            ctrl = np.array([ctrl(ob, i * env.dt)
                             for ctrl, ob in zip(ctrls, obs)])
            obs, rew, _, _ = env.step(ctrl)
            env.render()
        old_rew = indiv.fitness.values[0]
        diff = rew - old_rew
        perc = rew / old_rew if diff < 0. else old_rew / rew
        cprint("Reward: {:.3f} (difference: {:+.3f} - {:.1%})"
               .format(rew, diff, perc),
               'yellow')
    else:
        while True:
            env.render()
    if record:
        time.sleep(1)
        env.client.stopStateLogging(log_id)
    env.close()


def _format_indiv(indiv):
    """Helper function to create formated string to represent individuals"""
    movable = len([m for m in indiv.spawned_morph if m.joint])
    non_movable = len([m for m in indiv.spawned_morph if not m.joint])
    return 'Fitness: {:.3f} (Movable: {:2d}, Non: {:2d})'.format(
            indiv.fitness.values[0], movable, non_movable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morphology viewer")
    parser.add_argument('file', help="ZIP archive to load")
    parser.add_argument('--secs', type=float, default=20.,
                        help="Number of seconds to simulate for each")
    parser.add_argument('--max_size', type=int, default=20,
                        help="Maximum spawned morphology")
    parser.add_argument('--env', choices=['flat', 'ripple', 'ditch', 'square'],
                        default='flat', help="Environment to simulate")
    parser.add_argument('--checkpoint',
                        help="Load specific checkpoint")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--view', action='store_true',
                       help="Do not simulate control, only view")
    args = parser.parse_args()
    with zipfile.ZipFile(args.file, 'r') as archive:
        seeds = set(filter(lambda p: p and 'checkpoint' not in p,
                           map(os.path.dirname, archive.namelist())))
        seed = ""
        if len(seeds) > 1:
            cprint("Found {:d} seeds, please select:".format(len(seeds)), 'blue')
            for seed in sorted(seeds):
                print("{}".format(seed))
            seed = input("Select ID: ").strip()
        fil = [f for f in archive.namelist() if f.endswith('population.pickle')
               and seed in f and 'checkpoint' not in f]
        if fil and args.checkpoint is None:
            obj = pickle.loads(archive.read(fil[0]))
        else:
            ckpts = [f for f in archive.namelist()
                     if 'checkpoint' in f]
            if args.checkpoint is not None:
                fil = [f for f in ckpts if args.checkpoint in f][0]
            else:
                # Namelist is sorted by time so the last file will be the
                # latest checkpoint
                fil = [f for f in archive.namelist()
                       if 'checkpoint' in f][-1]
            cprint("Run did not finish, using checkpoint '{!s}'".format(fil),
                   'yellow')
            obj = pickle.loads(archive.read(fil))['population']
        m = Map((20, 20))
        for indiv in obj:
            behave = size_behavior(indiv, 20.)
            m.insert(behave, indiv)
        fitness = m.fitness()
        fitness = np.ma.masked_where(fitness < 0.001, fitness)
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        im = ax.imshow(fitness.T, origin='lower')
        plt.xlabel("Number of non-movable (except root module)")
        plt.ylabel("Number of movable")
        fig.colorbar(im)
        ax_gui = plt.axes([0.25, 0.01, 0.15, 0.1])
        gui_check = CheckButtons(ax_gui, ['GUI', 'Record'], actives=[1, 0])
        # Setup environment
        if args.env == 'flat':
            env = 'ModularLocomotion3D-v0'
        elif args.env == 'ripple':
            env = 'ModularRipple3D-v0'
        elif args.env == 'ditch':
            env = 'ModularDitch3D-v0'
        elif args.env == 'square':
            env = 'ModularSquareDitch3D-v0'

        # Helper method to handle mouse press
        def _on_click(event):
            if event.inaxes != ax:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            indiv = m._storage[x][y]
            checks = gui_check.get_status()
            if checks[0]:
                rec = 'recording.mp4' if checks[1] else None
                _evaluate(indiv, args.secs, args.max_size, args.view, rec, env)
            else:
                cprint("Testing: ({:d}, {:d})".format(x, y), 'green')
                rew, _ = evaluate(indiv, args.secs, args.max_size, 2.0, env)
                old_rew = indiv.fitness.values[0]
                diff = rew - old_rew
                perc = rew / old_rew if diff < 0. else old_rew / rew
                cprint("Reward: {:.3f} (difference: {:+.3f} - {:.1%})"
                       .format(rew, diff, perc),
                       'yellow')
        fig.canvas.mpl_connect('button_press_event', _on_click)
        plt.show()
