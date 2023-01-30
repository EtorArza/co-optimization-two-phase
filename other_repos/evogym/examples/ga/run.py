import os
import numpy as np
import shutil
import random
import math

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

PARALLEL = False

def run_ga(experiment_name, env_name, seed, max_evaluations, pop_size, structure_shape, num_cores, no: NestedOptimization):


    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    internal_exp_files = os.path.join(root_dir, "saved_data", experiment_name)
    temp_path = internal_exp_files + "/metadata.txt"

    default_train_iters = 1000
    default_num_steps = 128

    train_iters = int(no.inners_per_outer_proportion * default_train_iters)
    num_steps = int(no.inner_length_proportion * default_num_steps)



    tc = TerminationCondition(train_iters)
    tc_default = TerminationCondition(default_train_iters)

    if os.path.isdir(internal_exp_files):
        print("Replaceing internal exp. files:")
        import shutil
        shutil.rmtree(internal_exp_files)
 
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

    print("Start Optimization loop:")


    while True:

        ### UPDATE NUM SURVIORS ###			
        # We need to update the percentage survival based on the frames left and not on number of evaluations left.
        percent_survival = get_percent_survival_evals(no.step, no.max_frames) 
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        
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

        ### TRAIN GENERATION
        if PARALLEL:
            group = mp.Group()
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')
            else:

                if PARALLEL:
                    # For parallel execution
                    ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label), env_name, no, False)
                    group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
                else:
                    # For sequential execution
                    res = run_ppo((structure.body, structure.connections), tc, (save_path_controller, structure.label), env_name, no, False)
                    structure.set_reward(res)
                    if no.need_reevaluate:
                        no.sw.pause()
                        res_reevaluated = run_ppo((structure.body, structure.connections), tc_default, (save_path_controller, structure.label), env_name, no, True)
                        no.next_saverealobjective(res_reevaluated)
                        no.sw.resume()

        if PARALLEL:
            group.run_jobs(num_cores)


        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()


        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])

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