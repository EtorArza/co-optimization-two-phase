

"""
Evaluation functions for evolution
"""
import gym
import gym_rem
import numpy as np
import nevergrad as ng
from matplotlib import animation
from matplotlib import pyplot as plt
import os

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



import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _get_env(env='ModularLocomotion3D-v0'):
    return gym.make(env)



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


# def _morphology_tree_printer(node, max_depth, current_depth):

#     for child in node.children:
#         print((current_depth + 1)*"-", child)
#         if current_depth < max_depth:
#             _morphology_tree_printer(child,max_depth, current_depth+1)

def morphology_tree_printer(node, max_depth):
    stack = [(node, 0)]  # Start with the root node and depth 0
    while stack:
        current_node, current_depth = stack.pop()
        # print((current_depth + 1) * "-", current_node)
        if current_depth < max_depth:
            for child in reversed(current_node.children):
                stack.append((child, current_depth + 1))



def get_indiv_dim(individual, max_depth):
    node_count = 0
    controllable_node_count = 0
    stack = [(individual.morphology, 0)]  # Start with the root node and depth 0
    while stack:
        current_node, current_depth = stack.pop()
        if not current_node.joint is None: 
            controllable_node_count += 1
        node_count += 1
        if current_depth < max_depth:
            for child in reversed(current_node.children):
                stack.append((child, current_depth + 1))
    return node_count, controllable_node_count


def _copy_from_array_to_individual(value_list, individual, max_depth):

    controllable_node_idx = 0
    stack = [(individual.morphology, 0)]  # Start with the root node and depth 0
    while stack:
        current_node, current_depth = stack.pop()
        if not current_node.joint is None: 
            current_node.ctrl.amplitude = value_list[controllable_node_idx][0]
            current_node.ctrl.offset = value_list[controllable_node_idx][1]
            current_node.ctrl.frequency = value_list[controllable_node_idx][2]
            current_node.ctrl.phase = value_list[controllable_node_idx][3]
            controllable_node_idx += 1
        if current_depth < max_depth:
            for child in reversed(current_node.children):
                stack.append((child, current_depth + 1))
    assert len(value_list) == controllable_node_idx



def from_array_to_individual(array, individual, max_depth):
    from_01_to_range = lambda x, range: range[0] + x*(range[1]-range[0])
    from_range_to_01 = lambda x, range: (x - range[0]) / (range[1]-range[0])


    pi = 3.141592653589793238462643383279502884197169399375
    amp_range = (-1.57, 1.57)
    frequency_range = (0.2, 2.0)
    phase_range = (-2*pi, 2*pi)
    offset_range = (-1.57, 1.57)


    value_list = []
    assert len(array) % 4 == 0
    for idx in range(int(len(array) / 4)):
        value_list += [[from_01_to_range(array[idx*4+ 0], amp_range),
                                 from_01_to_range(array[idx*4+ 1], offset_range),
                                 from_01_to_range(array[idx*4+ 2], frequency_range),
                                 from_01_to_range(array[idx*4+ 3], phase_range),
                                 ]]

    _copy_from_array_to_individual(value_list, individual, max_depth)


def save_frames_as_gif(frames, save_animation_path):
    from matplotlib import pyplot as plt
    print("saving animation...", end="")


    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
   

    patch = plt.imshow(frames[0])
    
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=200)
    anim.save(save_animation_path, writer='imagemagick', fps=30)
    print("saved.")

def train_individual(individual, no=None, seconds=10.0, max_size=None, env='ModularLocomotion3D-v0', test=False):
    assert not no is None

    node_count, controllable_node_count = get_indiv_dim(individual, max_depth=4)

    # There is no need to simulated a morphology without joints
    if controllable_node_count == 0:
        return 0.0, None, controllable_node_count, node_count


    episode_budget = no.get_inner_quantity()
    assert episode_budget > 0 
    import warnings
    warnings.filterwarnings("ignore", message="DE algorithms are inefficient with budget < 60")
    parametrization = ng.p.Array(shape=(controllable_node_count*4,), lower=0.0, upper=1.0)
    parametrization.random_state.seed(no.get_seed())
    optimizer = ng.optimizers.TwoPointsDE(parametrization=parametrization, budget=episode_budget, num_workers=1)

    f_best = -1e100
    controler_best = None
    for _ in range(optimizer.budget):
        cand = optimizer.ask()
        controller = cand.value
        f = _evaluate_individual(individual, controller, no, seconds, max_size, env)[0]
        no.next_inner(f)
        if f > f_best:
            f_best = f
            controler_best = controller.copy()
        loss = -f
        optimizer.tell(cand, loss)

    return f_best, controler_best, controllable_node_count, node_count


def save_data_animation(dump_path, video_label, individual, controller, no, seconds, max_size, warmup, env):
    import pickle
    # assert len(controller) == len(ctrls)*4

    print("---save_data_animation()---")
    # morphology_tree_printer(individual.morphology, 4, 0)
    n_env = _get_env(env)
    obs = n_env.reset(morphology=individual.morphology, max_size=max_size)
    # morphology_tree_printer(n_env.morphology, 4, 0)
    # for m in n_env.morphology:
        # print(m,"->", m.joint, "->", m.ctrl if hasattr(m, "ctrl") else None)
    print("---save_data_animation()---")


    with open(dump_path, "wb") as f:
        pickle.dump((video_label, individual,controller,no, seconds, max_size, warmup, env), file=f)

# def morphology_tree_printer2(root, max_depth, current_depth):
#     print((current_depth + 1)*"-", root)
#     for child in root.children:
#         if current_depth < max_depth:
#             morphology_tree_printer2(child,max_depth, current_depth+1)
    

def animate_from_dump(dump_path):
    import pickle
    with open (dump_path, "rb") as f:
        video_label, individual,controller,no, seconds, max_size, warmup, env = pickle.load(f)
    no.params._inner_length_proportion = 1.0
    no.params._inner_quantity_proportion = 1.0
    print("---animate_from_dump()---")
    morphology_tree_printer(individual.morphology, 4)
    print("-end-")
    n_env = _get_env(env)
    obs = n_env.reset(morphology=individual.morphology, max_size=max_size)
    morphology_tree_printer(individual.morphology, 4)
    print("-end-")
    for m in n_env.morphology:
        print(m,"->", m.joint, "->", m.ctrl if hasattr(m, "ctrl") else None)
    print("---animate_from_dump()---")
    _evaluate_individual(individual, controller,no, seconds, max_size, env, save_animation = True, save_animation_path = f"results/jorgenrem/videos/{video_label}.gif")


def _evaluate_individual(individual, controller, no=None, seconds=10.0, max_size=None, env='ModularLocomotion3D-v0', save_animation=False, save_animation_path=None):
    """Evaluate the morphology in simulation"""
    assert not no is None


    from_array_to_individual(controller, individual, max_depth=4)

    warm_up = 0.0

    assert type(env) == str
    # hide diagnostic output
    with open(os.devnull, 'w') as devnull:
        # suppress stdout
        orig_stdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)
        # suppress stderr
        orig_stderr_fno = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), 2)

        env = gym.make(env)

        os.dup2(orig_stdout_fno, 1)  # restore stdout
        os.dup2(orig_stderr_fno, 2)  # restore stderr


    obs = env.reset(morphology=individual.morphology, max_size=max_size)
    import time
    time.sleep(1)
    ctrls = [m.ctrl for m in env.morphology if m.joint is not None]
    for m in env.morphology:
        if m.joint:
            m.ctrl.reset(m)
    steps = no.get_inner_length()

    if save_animation:
        assert not save_animation_path is None
        from tqdm import tqdm as tqdm
        pbar = tqdm(total=steps/4)

    warm_up = int(warm_up / env.dt)
    # Create copy to spawn in simulation
    obs = env.reset(morphology=individual.morphology, max_size=max_size)

    nnodes, ncontrolnodes = get_indiv_dim(individual, max_depth=4)

    assert len(controller) == ncontrolnodes*4
    if len(controller) == 0:
        return 0.0, env.morphology
    


    rew = 0.0
    warm_up_rew = 0.0
    # Step simulation until done
    frames = []
    for i in range(steps):
        obs = zip(*np.split(obs, 3))
        ctrl = np.array([ctrl(ob, i * env.dt) for ctrl, ob in zip(ctrls, obs)])
        obs, rew, _, _ = env.step(ctrl)

        if i <= warm_up:
            warm_up_rew = rew
        no.next_step()
        if save_animation and i % 4 == 0:
            pbar.update(1)
            frame = env.render("rgb_array")
            frames.append(frame)




    f = rew - warm_up_rew
    no.next_inner(f_partial=f)

    if save_animation:
        save_frames_as_gif(frames, save_animation_path=save_animation_path)


    return rew - warm_up_rew, env.morphology


def evaluate(individual, no=None, seconds=10.0, max_size=None, warm_up=0.0, env='ModularLocomotion3D-v0'):
    

    f_og, _, controller_len, morph_size = train_individual(individual, no, seconds, max_size, env, test=False)

    no.next_outer(f_og, controller_len, -1, morph_size)
    if no.is_reevaluating_flag:
        f_reeval, controller_best,  _, _ = train_individual(individual, no, seconds, max_size, env, test=True)
        no.next_reeval(f_reeval, controller_len, -1, morph_size)
        print(f"Save current animation with f_reeval={f_reeval}!")
        save_data_animation(f"dumps_for_animation/animation_dump_current{no.params.experiment_index}.wb", f"vid_{no.get_video_label()}_current", individual, controller_best, no, seconds, max_size, warm_up, env)

        if no.new_best_found:
            print(f"Save best animation with f_reeval={f_reeval}!")
            save_data_animation(f"dumps_for_animation/animation_dump_best{no.params.experiment_index}.wb", f"vid_{no.get_video_label()}_best", individual, controller_best, no, seconds, max_size, warm_up, env)
            no.new_best_found = False
    return f_og, individual.morphology










    print("Training individual", individual)
    return f_og, morph_og