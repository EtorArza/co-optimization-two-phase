#!/usr/bin/env python

"""
Evaluation functions for evolution
"""
import gym
import gym_rem
import numpy as np


def load(toolbox, settings):
    """Load evaluation function into toolbox"""
    secs = settings.getfloat('evaluation', 'time')
    size = settings.getint('morphology', 'max_size')
    warm_up = 0.0
    if 'warm_up' in settings['evaluation']:
        warm_up = settings.getfloat('evaluation', 'warm_up')
    enviro = 'ModularLocomotion3D-v0'
    if 'environment' in settings['evaluation']:
        enviro = settings['evaluation']['environment']
        assert gym.spec(enviro) in gym.envs.registry.all(), "Unknown environment '{!s}'!".format(enviro)
    toolbox.register('evaluate', evaluate, seconds=secs, max_size=size,
                     warm_up=warm_up, env=enviro)


# Global variable to persist simulation environment between calls
SIM_ENV = None


def _get_env(env='ModularLocomotion3D-v0'):
    """Initialize modular environment"""
    global SIM_ENV
    if SIM_ENV is None:
        SIM_ENV = gym.make(env)
    return SIM_ENV


def evaluate(individual, seconds=10.0, max_size=None, warm_up=0.0,
             env='ModularLocomotion3D-v0'):
    """Evaluate the morphology in simulation"""
    env = _get_env(env)
    steps = int(seconds / env.dt)
    warm_up = int(warm_up / env.dt)
    # Create copy to spawn in simulation
    obs = env.reset(individual.morphology, max_size=max_size)
    ctrls = [m.ctrl for m in env.morphology if m.joint is not None]
    # There is no need to simulated a morphology without joints
    if not ctrls:
        return 0.0, env.morphology
    # Tell controller nodes that morphology is final
    for m in env.morphology:
        if m.joint:
            m.ctrl.reset(m)
    rew = 0.0
    warm_up_rew = 0.0
    # Step simulation until done
    for i in range(steps):
        obs = zip(*np.split(obs, 3))
        ctrl = np.array([ctrl(ob, i * env.dt) for ctrl, ob in zip(ctrls, obs)])
        obs, rew, _, _ = env.step(ctrl)
        if i <= warm_up:
            warm_up_rew = rew
    return rew - warm_up_rew, env.morphology
