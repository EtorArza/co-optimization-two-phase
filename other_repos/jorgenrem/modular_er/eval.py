

"""
Evaluation functions for evolution
"""
import gym
import gym_rem
import numpy as np
import nevergrad as ng


def load(toolbox, settings, no):
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
    toolbox.register('evaluate', evaluate, no=no, seconds=secs, max_size=size,
                     warm_up=warm_up, env=enviro)


# Global variable to persist simulation environment between calls
SIM_ENV = None


def _get_env(env='ModularLocomotion3D-v0'):
    """Initialize modular environment"""
    global SIM_ENV
    if SIM_ENV is None:
        SIM_ENV = gym.make(env)
    return SIM_ENV



def from_ctrls_to_array(ctrls):

    from_01_to_range = lambda x, range: range[0] + x*(range[1]-range[0])
    from_range_to_01 = lambda x, range: (x - range[0]) / (range[1]-range[0])


    pi = 3.141592653589793238462643383279502884197169399375
    amp_range = (-1.57, 1.57)
    frequency_range = (0.2, 2.0)
    phase_range = (-2*pi, 2*pi)
    offset_range = (-1.57, 1.57)

    res = np.zeros(4*len(ctrls))


    for idx in range(len(ctrls)):
        res[idx*4+ 0] = from_range_to_01(ctrls[idx].amplitude, amp_range)
        res[idx*4+ 2] = from_range_to_01(ctrls[idx].frequency, frequency_range)
        res[idx*4+ 3] = from_range_to_01(ctrls[idx].phase, phase_range)
        res[idx*4+ 1] = from_range_to_01(ctrls[idx].offset, offset_range)


    return res





def from_array_to_ctrls(array, ctrls):
    from_01_to_range = lambda x, range: range[0] + x*(range[1]-range[0])
    from_range_to_01 = lambda x, range: (x - range[0]) / (range[1]-range[0])

    assert len(array) == len(ctrls)*4

    pi = 3.141592653589793238462643383279502884197169399375
    amp_range = (-1.57, 1.57)
    frequency_range = (0.2, 2.0)
    phase_range = (-2*pi, 2*pi)
    offset_range = (-1.57, 1.57)

    
    for idx in range(len(ctrls)):
        ctrls[idx].amplitude = from_01_to_range(array[idx*4+ 0], amp_range)
        ctrls[idx].frequency = from_01_to_range(array[idx*4+ 2], frequency_range)
        ctrls[idx].phase = from_01_to_range(array[idx*4+ 3], phase_range)
        ctrls[idx].offset = from_01_to_range(array[idx*4+ 1], offset_range)


def train_individual(individual, no=None, seconds=10.0, max_size=None, warm_up=0.0, env='ModularLocomotion3D-v0'):
    assert not no is None
    env = _get_env(env)
    warm_up = int(warm_up / env.dt)
    # Create copy to spawn in simulation
    obs = env.reset(morphology=individual.morphology, max_size=max_size)
    ctrls = [m.ctrl for m in env.morphology if m.joint is not None]
    # There is no need to simulated a morphology without joints
    if not ctrls:
        return 0.0, env.morphology
    
    n = len(from_ctrls_to_array(ctrls))



    episode_budget = no.get_inner_quantity()
    parametrization = ng.p.Array(shape=(n,), lower=0.0, upper=1.0)
    parametrization.random_state.seed(no.get_seed())
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=episode_budget, num_workers=1)

    f_best = -1e100
    controler_best = None
    for _ in range(optimizer.budget):
        cand = optimizer.ask()
        controller = cand.value
        f = _evaluate_individual(individual, controller, no, seconds, max_size, warm_up, env)[0]
        no.next_inner(f)
        if f > f_best:
            f_best = f
            controler_best = controller.copy()
        loss = -f
        optimizer.tell(cand, loss)


    no.next_outer(f, len(controller), -1, len(individual.morphology))
    return f_best, env.morphology






def _evaluate_individual(individual, controller, no=None, seconds=10.0, max_size=None, warm_up=0.0, env='ModularLocomotion3D-v0'):
    """Evaluate the morphology in simulation"""
    assert not no is None

    warm_up = 0.0

    env = _get_env(env)
    steps = no.get_inner_length()
    warm_up = int(warm_up / env.dt)
    # Create copy to spawn in simulation
    obs = env.reset(morphology=individual.morphology, max_size=max_size)
    ctrls = [m.ctrl for m in env.morphology if m.joint is not None]
    # There is no need to simulated a morphology without joints
    if not ctrls:
        return 0.0, env.morphology
    

    from_array_to_ctrls(controller, ctrls)
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
        no.next_step()

    f = rew - warm_up_rew
    no.next_inner(f_partial=f)
    print(ctrls, f)

    return rew - warm_up_rew, env.morphology


def evaluate(individual, no=None, seconds=10.0, max_size=None, warm_up=0.0, env='ModularLocomotion3D-v0'):
    return train_individual(individual, no, seconds, max_size, warm_up, env)