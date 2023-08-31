import os, sys, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "rlkit"))
sys.path.append(os.path.join(os.path.dirname(__file__)))

import hashlib
import coadapt
import experiment_configs as cfg
import json
from NestedOptimization import NestedOptimization


# Params explained. From the paper:

"""
How it works in the paper by default:

- Per episode we train the individual networks pi_ind. and q_ind. 1000 times ['indiv_updates': 1000],
while the population networks pi_pop. and q_pop. are trained 250 times ['pop_updates': 250].
- A batch size of 256 was used for each training updated ['batch_size': 256].
- config["iterations_random"] does not matter. Its only for some baseline experiments.
- For the standard PyBullet tasks we executed 300 episodes ['iterations_init': 300] for the initial five 
designs and 100 episodes ['iterations': 100] thereafter.
- config["design_cycles"] is the number of designs as stopping criterion. In our case, it should be +inf
as the stopping criterion is based on the number of frames.

How we interpreted the parameters for our framework:


"""



def main(config, no: NestedOptimization):
    config["data_folder"] = "other_repos/kevincoadapting/data_exp_sac_pso_batch"
    folder = config['data_folder']

    params="toy"

    if params=="actual":
        # Actual params
        config["steps_per_episode"] = int(float(no.get_inner_length()) / 100.0 * config["steps_per_episode"])
        config["state_batch_size"] = int(float(no.get_inner_length()) / 100.0 * config["state_batch_size"])

        config["rl_algorithm_config"]["pop_updates"] = int(float(no.get_inner_length()) / 100.0 * config["rl_algorithm_config"]["pop_updates"])
        config["rl_algorithm_config"]["indiv_updates"] = int(float(no.get_inner_length()) / 100.0 * config["rl_algorithm_config"]["indiv_updates"])

        config["iterations_init"] = 2
        config["iterations"] = 3


        config["design_cycles"] = 99999999999



    elif params=="toy":
        # Toy params
        config["steps_per_episode"] = 10
        config["state_batch_size"] = 4

        config["rl_algorithm_config"]["pop_updates"] = 2
        config["rl_algorithm_config"]["indiv_updates"] = 2
        config["design_cycles"] = 3

        config["iterations_init"] = 8
        config["iterations"] = 3



    # config["iterations"] = 2
    # config["iterations_random"] = 2
    # config["iterations_init"] = 2
    # config["rl_algorithm_config"]["pop_updates"] = 2
    # config["rl_algorithm_config"]["indiv_updates"] = 2



    #generate random hash string - unique identifier if we start
    # multiple experiments at the same time
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id
    config['data_folder_experiment'] = file_str

    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    # Store config
    with open(os.path.join(file_str, 'config.json'), 'w') as fd:
            fd.write(json.dumps(config,indent=2))

    co = coadapt.Coadaptation(config)
    # import cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()
    co.run()
    # profiler.disable()
    # profiler.dump_stats("profile_data.prof")


if __name__ == "__main__":
    # We assume we call the program only with the name of the config we want to run
    # nothing too complex
    if len(sys.argv) > 1:
        config = cfg.config_dict[sys.argv[1]]
    else:
        config = cfg.config_dict['base']
    main(config)
