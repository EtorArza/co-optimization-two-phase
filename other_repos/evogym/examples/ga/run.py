import os
import numpy as np
import shutil
import random
import math
import pathlib

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from NestedOptimization import NestedOptimization

from ppo.envs import make_vec_envs
import torch
from ppo.utils import get_vec_normalize
import imageio
from pygifsicle import optimize
import pickle


def dump_visualization_data(dump_path, out_gif_path, env_name, structure, ctrl_path):


    out_gif_path, env_name, structure, ctrl_path

    print("Saving simulation objects...", end="")

    # an example object to be pickled
    simulation_objects = [out_gif_path, env_name, structure, ctrl_path]
    print("ctrl_path =",ctrl_path)
    # pickling the object
    with open(dump_path, "wb") as f:
        pickle.dump(simulation_objects, f)
    print("done.")


def load_visualization_data(pickle_dump_path):
    with open(pickle_dump_path, "rb") as f:
        out_path, env_name, structure, ctrl_path = pickle.load(f)

  
    return out_path, env_name, structure, ctrl_path

def save_robot_gif_standalone(out_path, env_name, structure, ctrl_path):
    print("Generating gif...")
    gif_resolution = (1280/5, 720/5)

    env = make_vec_envs(env_name, structure, 1000, 1, None, None, device='cpu', allow_early_resets=False)
    env.get_attr("default_viewer", indices=None)[0].set_resolution(gif_resolution)
                    
    actor_critic, obs_rms = torch.load(ctrl_path, map_location='cpu')

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()
    img = env.render(mode='img')
    reward = None
    done = False

    imgs = []
    # arrays = []
    while not done:

        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)

        obs, reward, done, _ = env.step(action)
        img = env.render(mode='img')
        imgs.append(img)

        masks.fill_(0.0 if (done) else 1.0)

        if done == True:
            env.reset()

    env.close()
    imageio.mimsave(out_path, imgs, duration=(1/50.0))



def run_ga(pop_size, structure_shape, no: NestedOptimization):

    seed = no.seed
    env_name = no.env_name
    max_evaluations = no.max_frames

    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    internal_exp_files = os.path.join(root_dir, "saved_data", f"exp{no.experiment_index}")
    temp_path = internal_exp_files + "/metadata.txt"

    # # To test stuff. If default_train_iters < 100 it does not work. 
    # # This is because the performance of the model is only saved every 50 iterations, and 
    # # if the parameter inner inners_per_outer_proportion is 0.5, we get 50 iterations when  
    # # default_train_iters = 100.
    # default_train_iters = 100 

    default_train_iters = 100
    default_num_steps = 12

    train_iters = int(no.inners_per_outer_proportion * default_train_iters)
    num_steps = int(no.inner_length_proportion * default_num_steps)



    tc = TerminationCondition(train_iters)
    tc_default = TerminationCondition(default_train_iters)

    if os.path.isdir(internal_exp_files):
        print("Removing old exp. files:")
        import shutil
        shutil.rmtree(internal_exp_files, ignore_errors=True)

    os.makedirs(internal_exp_files)


    f = open(temp_path, "w")
    f.write(f'POP_SIZE: {pop_size}\n')
    f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
    f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
    f.write(f'TRAIN_ITERS: {train_iters}\n')
    f.close()



    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0
    
    #generate a population
    for i in range (pop_size):
        
        temp_structure = sample_robot(structure_shape)
        while (hashable(temp_structure[0]) in population_structure_hashes):
            temp_structure = sample_robot(structure_shape)

        structures.append(Structure(*temp_structure, i))
        population_structure_hashes[hashable(temp_structure[0])] = True
        num_evaluations += 1



    while True:

        ### UPDATE NUM SURVIORS ###			
        # We need to update the percentage survival based on the frames left and not on number of evaluations left.
        percent_survival = get_percent_survival_evals(no.step, no.max_frames) 
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", f"exp{no.experiment_index}", "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", f"exp{no.experiment_index}", "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", f"exp{no.experiment_index}", "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", f"exp{no.experiment_index}", "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')
            else:

                # For sequential execution
                res = run_ppo((structure.body, structure.connections), tc, (save_path_controller, structure.label), env_name, no, False)
                structure.set_reward(res)


                if no.is_reevaluating:
                    controller_path_for_animation_current = f"controller_to_generate_animation_{no.experiment_index}_current.pt"
                    controller_path_for_animation_best = f"controller_to_generate_animation_{no.experiment_index}_best.pt"
                    no.controller_path_for_animation = controller_path_for_animation_current
                    res_reevaluated = run_ppo((structure.body, structure.connections), tc_default, (save_path_controller, structure.label), env_name, no, True)
                    
                    import pathlib

                    dump_path_current = f"simulation_objects_{no.experiment_index}_current.pkl"
                    dump_path_best = f"simulation_objects_{no.experiment_index}_best.pkl"


                    morphology = structure.body
                    controller_size = np.sum(morphology == 3) + np.sum(morphology == 4)
                    controller_size2 = structure.connections.shape[1]
                    morphology_size = np.sum(morphology != 0)
                    no.next_reeval(res_reevaluated, controller_size, controller_size2, morphology_size)
                    out_path_gif_current = pathlib.Path().resolve().as_posix() + f"../../../../results/evogym/videos/vid_{no.get_video_label()}_current.gif"
                    out_path_gif_best = pathlib.Path().resolve().as_posix() + f"../../../../results/evogym/videos/vid_{no.get_video_label()}_best.gif"                    
                    dump_visualization_data(dump_path_current, out_path_gif_current, env_name, (structure.body, structure.connections), controller_path_for_animation_current)

                    if no.save_best_visualization_required:
                        no.save_best_visualization_required = False
                        os.system(f"cp {controller_path_for_animation_current} {controller_path_for_animation_best}")

                        dump_visualization_data(dump_path_best, out_path_gif_best, env_name, (structure.body, structure.connections), controller_path_for_animation_best)
                        
                    

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", no.experiment_index, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()


        # print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        # print(structures[:num_survivors])

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]


        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]

        generation += 1