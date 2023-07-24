

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
import copy
from collections import deque

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


def _get_collision_hash(morphology):
    

    env = _get_env('ModularLocomotion3D-v0')
    env.morphology = copy.deepcopy(morphology.root)
    # Mapping from module to PyBullet ID
    spawned_ids = {}
    # NOTE: We are using explicit queue handling here so that we can
    # ignore children of overlapping modules
    queue = deque([env.morphology.root])
    max_size = 4

    aabb_list = []

    while len(queue) > 0:
        module = queue.popleft()
        # Spawn module in world
        m_id = module.spawn(env.client)
        # Check if the module overlaps
        aabb_min, aabb_max = env.client.getAABB(m_id)
        # Check overlapping modules
        overlap = env.client.getOverlappingObjects(aabb_min, aabb_max)
        # NOTE: An object always collides with it env
        overlapping_modules = False
        if overlap:
            overlapping_modules = any([u_id != m_id for u_id, _ in overlap])
        # Check against plane
        aabb_min, aabb_max = env.client.getAABB(env.plane_id)
        aabb_list += [aabb_min,aabb_max]
        plane_overlap = env.client.getOverlappingObjects(aabb_min, aabb_max)
        overlapping_plane = False
        if plane_overlap:
            overlapping_plane = any([u_id != env.plane_id
                                        for u_id, _ in plane_overlap])
        # If overlap is detected de-spawn module and continue
        if overlapping_modules or overlapping_plane:
            # Remove from simulation
            env.client.removeBody(m_id)
            # Remove from our private copy
            parent = module.parent
            if parent:
                del parent[module]
            else:
                raise RuntimeError("Trying to remove root module due to collision!")
            continue
        # Add children to queue for processing
        queue.extend(module.children)
        # Add ID to spawned IDs so that we can remove them later
        spawned_ids[module] = m_id
        # Check size constraints
        if max_size is not None and len(spawned_ids) >= max_size:
            # If we are above max desired spawn size drain queue and remove
            for module in queue:
                parent = module.parent
                if parent:
                    del parent[module]
                else:
                    raise RuntimeError("Trying to prune root link!")
            break
    return str(aabb_list)


def _morphology_string(node, max_depth):
    res = []
    stack = [(node, 0)]  # Start with the root node and depth 0
    while stack:
        current_node, current_depth = stack.pop()
        res.append(str(current_depth) + "_" + str(current_node))
        if current_depth < max_depth:
            for child in reversed(current_node.children):
                stack.append((child, current_depth + 1))
    return res

def get_morphology_hash(morphology):
    env_name = 'ModularLocomotion3D-v0'
    max_depth = 4
    env = _get_env(env_name)
    env.reset(morphology=morphology, max_size=max_depth)
    m_id = env.morphology.root.spawn(env.client)
    aabb_min, aabb_max = env.client.getAABB(m_id)
    hash = "|".join([str(el) for el in [aabb_min, aabb_max, env.morphology.root.position, _morphology_string(env.morphology.root,max_depth), _get_collision_hash(morphology)]])        
    env.client.removeBody(m_id)
    return hash


import sys
from contextlib import contextmanager
import ctypes


class RedirectStream(object):

  @staticmethod
  def _flush_c_stream(stream):
    streamname = stream.name[1:-1]
    libc = ctypes.CDLL(None)
    libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

  def __init__(self, stream=sys.stdout, file=os.devnull):
    self.stream = stream
    self.file = file

  def __enter__(self):
    self.stream.flush()  # ensures python stream unaffected 
    self.fd = open(self.file, "w+")
    self.dup_stream = os.dup(self.stream.fileno())
    os.dup2(self.fd.fileno(), self.stream.fileno()) # replaces stream
  
  def __exit__(self, type, value, traceback):
    RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
    os.dup2(self.dup_stream, self.stream.fileno()) # restores stream
    os.close(self.dup_stream)
    self.fd.close()



def _get_env(env='ModularLocomotion3D-v0'):
    with RedirectStream(sys.stdout):
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
        return 0.0, None, controllable_node_count, node_count, None


    episode_budget = no.get_inner_quantity()
    assert episode_budget > 0 
    import warnings
    warnings.filterwarnings("ignore", message="DE algorithms are inefficient with budget < 60")
    parametrization = ng.p.Array(shape=(controllable_node_count*4,), lower=0.0, upper=1.0)
    parametrization.random_state.seed(no.get_seed())
    optimizer = ng.optimizers.TwoPointsDE(parametrization=parametrization, budget=episode_budget, num_workers=1)

    f_best = -1e100
    evaluation_signature_best = None
    controler_best = None
    for _ in range(optimizer.budget):
        cand = optimizer.ask()
        controller = cand.value
        f, _, evaluation_signature = _evaluate_individual(individual, controller, no, seconds, max_size, env)

        no.next_inner(f)
        if f > f_best:
            f_best = f
            controler_best = controller.copy()
            evaluation_signature_best = evaluation_signature
        loss = -f
        optimizer.tell(cand, loss)

    return f_best, controler_best, controllable_node_count, node_count, evaluation_signature_best


def save_data_animation(dump_path, video_label, individual, controller, no, seconds, max_size, env, evaluation_signature):
    import pickle
    # assert len(controller) == len(ctrls)*4

    n_env = _get_env(env)
    obs = n_env.reset(morphology=individual.morphology, max_size=max_size)


    with open(dump_path, "wb") as f:
        pickle.dump((video_label, individual,controller,no, seconds, max_size, env, evaluation_signature), file=f)



def animate_from_dump(dump_path):
    import pickle
    with open (dump_path, "rb") as f:
        video_label, individual,controller,no, seconds, max_size, env, saved_evaluation_signature = pickle.load(f)
    no.params._inner_length_proportion = 1.0
    no.params._inner_quantity_proportion = 1.0
    # n_env = _get_env(env)
    # obs = n_env.reset(morphology=individual.morphology, max_size=max_size)
    # print("Individual dims:", get_indiv_dim(individual, 4))
    _,_,evaluation_signature = _evaluate_individual(individual, controller,no, seconds, max_size, env, save_animation = True, save_animation_path = f"results/jorgenrem/videos/{video_label}.gif")


    f = float(evaluation_signature.split("||f_final:")[-1])

    f_saved = float(saved_evaluation_signature.split("||f_final:")[-1])





    if saved_evaluation_signature == evaluation_signature:
        print("Evaluation signatures match.")
    else:
        print("f during optimization:", f)
        print("f on animation:", f_saved)
        print("deviation: ", abs(f-f_saved) / max(f,f_saved))
        print("Evaluation signatures dont match.")
        print("ESignature dump:")
        print("---")
        print(saved_evaluation_signature)
        print("---")
        print("ESignature current:")
        print(evaluation_signature)
        print("---")
        exit(1)

def _evaluate_individual(individual, controller, no=None, seconds=10.0, max_size=None, env='ModularLocomotion3D-v0', save_animation=False, save_animation_path=None):
    """Evaluate the morphology in simulation"""
    assert not no is None


    from_array_to_individual(controller, individual, max_depth=4)

    warm_up = 0.0

    assert type(env) == str

    env = _get_env()



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

    nnodes, ncontrolnodes = get_indiv_dim(individual, max_depth=4)

    assert len(controller) == ncontrolnodes*4
    if len(controller) == 0:
        return 0.0, env.morphology, None
    


    rew = 0.0
    warm_up_rew = 0.0
    # Step simulation until done
    frames = []
    observed_scores = []
    for i in range(steps):
        obs = zip(*np.split(obs, 3))
        ctrl = np.array([ctrl(ob, i * env.dt) for ctrl, ob in zip(ctrls, obs)])
        
        if i > steps - 5:
            observed_scores += [str((ctrl))]
        obs, rew, _, _ = env.step(ctrl)

        if i <= warm_up:
            warm_up_rew = rew
        no.next_step()
        if save_animation and i % 4 == 0:
            pbar.update(1)
            frame = env.render("rgb_array")
            frames.append(frame)

    observed_scores = str(observed_scores)
    morph_hash = get_morphology_hash(individual.morphology)

    evaluation_signature = "observedscores:"+observed_scores+"||morph_hash:"+morph_hash

    f = rew - warm_up_rew
    no.next_inner(f_partial=f)

    if save_animation:
        save_frames_as_gif(frames, save_animation_path=save_animation_path)


    return rew - warm_up_rew, env.morphology, evaluation_signature+"||f_final:"+str(f)


def evaluate(individual, no=None, seconds=10.0, max_size=None, warm_up=0.0, env='ModularLocomotion3D-v0'):
    

    f_og, _, controller_len, morph_size, evaluation_signature = train_individual(individual, no, seconds, max_size, env, test=False)

    no.next_outer(f_og, controller_len, -1, morph_size)
    if no.is_reevaluating_flag:
        f_reeval, controller_best,  _, _ , evaluation_signature_best = train_individual(individual, no, seconds, max_size, env, test=True)
        no.next_reeval(f_reeval, controller_len, -1, morph_size)
        print(f"Save current animation with f_reeval={f_reeval}!")
        save_data_animation(f"dumps_for_animation/animation_dump_current{no.params.experiment_index}.wb", f"vid_{no.get_video_label()}_current", individual, controller_best, no, seconds, max_size, env, evaluation_signature_best)

        if no.new_best_found:
            print(f"Save best animation with f_reeval={f_reeval}!")
            save_data_animation(f"dumps_for_animation/animation_dump_best{no.params.experiment_index}.wb", f"vid_{no.get_video_label()}_best", individual, controller_best, no, seconds, max_size, env, evaluation_signature_best)
            no.new_best_found = False
    return f_og, individual.morphology










    print("Training individual", individual)
    return f_og, morph_og