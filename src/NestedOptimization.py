import time
from threading import Lock
import numpy as np
from datetime import datetime
import os
import functools
from pathlib import Path


# https://stackoverflow.com/a/32238541/13012332
def monitor_results(func):
    @functools.wraps(func)
    def wrapper(*func_args, **func_kwargs):
        print('function call ' + func.__name__ + '()')
        retval = func(*func_args,**func_kwargs)
        print('function ' + func.__name__ + '() returns ' + repr(retval))
        return retval
    return wrapper



def convert_from_seconds(seconds):


    mins, _ = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)

    # Format the result as a string
    result = f"{days} d, {hours} h, {mins} min"
    return result


class stopwatch:
    paused=False
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0
        self.paused=False

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        # print("Pause time = ", self.pause_t)
        # print("Time without pause = ", time.time() - self.start_t)
        # print("Time = ", time.time() - self.start_t - self.pause_t)
        current_extra_pause_time = 0.0
        if self.paused:
            current_extra_pause_time = time.time() - self.pause_start
        return time.time() - self.start_t - self.pause_t - current_extra_pause_time

    def get_time_string_short_format(self):
        return "{:.4f}".format(self.get_time())

class Parameters:
    nseeds = 60
    nseeds_proposed_method = 4

    def __init__(self, framework_name: str, experiment_index: int):

        self.experiment_index = experiment_index
        self.framework_name = framework_name
        self.reevaluated_was_new_best_flags = []
        if framework_name == "evogym":

            # If default_train_iters < 100 (to test stuff) it does not work. This is because the performance
            # of the model is only saved every 50 iterations, and if the parameter inner 
            # _inner_quantity_proportion is 0.5, we get 50 iterations when default_train_iters = 100.

            self.max_frames = 8633000 # Default 34532000 considering 250 morphologies evaluated (easy tasks).
            self.env_name_list = ["Walker-v0"]
            self._default_inner_quantity = 1000 # The number of the iterations of the PPO algorithm.
            self._default_inner_length = 128 # Episode length.
            self.non_resumable_param = "length"


        elif framework_name == "robogrammar":
            self.max_frames = 10240000 # max_frames=40960000 is the default value if we consider 5000 iterations as in the example in the GitHub.
            self.env_name_list = ["FlatTerrainTask"]
            self._default_inner_quantity = 64 # The number of random samples to generate when simulating each step.
            self._default_inner_length = 128 # Episode length.
            self.non_resumable_param = "quantity"


        elif framework_name == "tholiao":
            self.max_frames = 500000 # max_frames=2000000 is the default value (total steps)
            self.env_name_list = ["FlatTerrainTask"]
            self._default_inner_quantity =  50 # The number of controllers tested per morphology
            self._default_inner_length = 400 # Steps per episode
            self.non_resumable_param = "length"

        elif framework_name == "gymrem2d":
            # max_frames=3002177 is the default value on average. However, the framework 
            # itself is not a nested optimization approach. The framework by default evaluates 10000.
            # using 3002177 frames would not be enough because each morphology is now evaluated multiple times.
            # hence, we consider a budget x4 of the original, 

            # self.max_frames = 12008708
            self.max_frames = 12008708

            self.env_name_list = ["default"]
            self._default_inner_quantity =  64 # <- Need to do some parameter search!!!
            self._default_inner_length = 424 # Represents the average episode length in steps
            self.non_resumable_param = "length"


        elif framework_name == "jorgenrem":
            self.max_frames = 480000000 # max_frames=480000000 is the default value
            self.env_name_list = ["default"]
            self._default_inner_quantity =  10 # <- Need to do some parameter search!!!
            self._default_inner_length = 4800 # Represents the number of steps per controller tested
            self.non_resumable_param = "length"

        else:
            raise ValueError(f"Framework {framework_name} not found.")


        params = self._get_parameter_list()[experiment_index]

        if "reevaleachvsend" in params:
            self.experiment_mode, self.seed, self._inner_quantity_proportion, self._inner_length_proportion, self.env_name = params
        elif "standard" in params or "ctalatii" in params or "gesp" in params or "ctalatii&gesp" in params:
            self._inner_quantity_proportion = 1.0
            self._inner_length_proportion = 1.0
            self.method_mode, self.seed, self.env_name,  = params
            self.experiment_mode = "proposedmethod"

        elif "adaptstepspermorphology" in params:
            print("With the new experiment, the objective value used as a reference for wether to reevaluate is generated with different inner_quantity parameter from the new solutions. This is a problem, because if the inner quantity is smaller, it means that objective funcions in general will be lower, and it would be less likely to find new best solutions. This can be solved in evogym because we generate partial objective values with different inner quantity values, but it might not be possible to do in robogrammar. A possibility is to partially reevaluate best solution and thus get a valid reference.")
            print("Exiting...")
            exit(0)
            self.target_probability, self._start_quantity_proportion, self.env_name, self.experiment_mode, self.seed = params
            self._inner_quantity_proportion = self._start_quantity_proportion
        else:
            raise ValueError(f"params variable {params} does not contain a recognized experiment")


    def _get_inner_quantity_absolute(self):
            return int(self._inner_quantity_proportion * self._default_inner_quantity)

    def _get_inner_length_absolute(self):
        return int(self._inner_length_proportion * self._default_inner_length)


    def _get_parameter_list(self):
        import itertools
        res = []
        seed_list = list(range(1,1 + self.nseeds))

        # reevaleachvsend
        _inner_quantity_proportion_list = [0.1, 0.25, 0.5, 0.75, 1.0]
        _inner_length_proportion_list =   [0.1, 0.25, 0.5, 0.75, 1.0]
        experiment_mode_list = ["reevaleachvsend"]
        params_with_undesired_combinations = list(itertools.product(experiment_mode_list, seed_list, _inner_quantity_proportion_list, _inner_length_proportion_list, self.env_name_list))
        params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item[2:4]] # remove the combinations in which one of the parameters is not 1.0.
        res += params_with_undesired_combinations


        # proposedmethod
        seed_list = list(range(1,1 + self.nseeds_proposed_method))
        # "ctalatii" = Continue training as long as there is improvement.
        # "gesp"
        self.ctalatii_reference_ratio = 0.5 # max time until stop with ctalatii, where 0.5 means in the last half of the evaluations no improvement observed.
        self.max_time_per_morph_ratio = 0.05 # max time to spend per morphology divided by max time total.
        self.esnof_min_time_ratio = 0.01 * self.max_time_per_morph_ratio # also grace time

        _method_list = ["standard", "ctalatii", "gesp", "ctalatii&gesp"] 
        params = list(itertools.product(_method_list, seed_list, self.env_name_list))
        params = [item for item in params]
        res += params

        # # adaptstepspermorphology
        # target_probability = [0.75]
        # _start_quantity_proportion = [0.1]
        # experiment_mode_list = ["adaptstepspermorphology"]
        # params_with_undesired_combinations = list(itertools.product(target_probability, _start_quantity_proportion, self.env_name_list, experiment_mode_list, seed_list))
        # res += params_with_undesired_combinations

        return res



    def get_result_file_name(self):
        if self.experiment_mode == "reevaleachvsend":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self._inner_quantity_proportion}_{self._inner_length_proportion}_{self.seed}"
        if self.experiment_mode == "proposedmethod":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self.method_mode}_{self.seed}"
        if self.experiment_mode == "adaptstepspermorphology":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self.target_probability}_{self._start_quantity_proportion}_{self.seed}"
        else:
            raise NotImplementedError(f"Result file name not implemented for {self.experiment_mode} yet.")

    def get_n_experiments(self):
        return len(self._get_parameter_list())

    def print_parameters(self):
        print("Total number of executions:", self.get_n_experiments())

        # Create a dictionary to store the positions for each element
        positions = {}

        # Iterate through the list of tuples
        for i, key_tup in enumerate(self._get_parameter_list()):
            key = key_tup[0]
            if key not in positions:
                positions[key] = {"start": i, "end": i}
            else:
                positions[key]["end"] = i

        # Print the table
        for key, info in positions.items():
            start = info["start"]
            end = info["end"]
            if start == end:
                print(f"{key}, {start}")
            else:
                print(f"{key}, {start}-{end}")

        print("Total number of executions:", )

        print("Parameters current execution:",self._get_parameter_list()[self.experiment_index])





class NestedOptimization:

    """
    Experiment modes:

    ----------------------------------
    reevaleachvsend
    
    We try to find the best morphology with a reduced amount of resources.
    Once we have found the best candidate morphology, we retrain this morphology with proper resources.
    This 'retraining' can be done each time a new best morphology is found, or, at the end of the search.
    ----------------------------------
    proposedmethod

    We apply ESNOF in the 'resumable' parameter, and we incrementally increase the other parameter throughout
    the execution. 
    ----------------------------------
    adaptstepspermorphology
    
    Adapt the steps used in training each morphology such that probability of finding a new 'best' morphology 
    that is not actually best when properly trained stays low. At the same time, we want to minimize the steps
    used to train each morphology.
    ----------------------------------
    


    """
    sw_print_progress = stopwatch()
    sw = stopwatch()
    sw_reeval = stopwatch()
    f_observed = float("-inf")
    f_best = float("-inf")
    f_reeval_observed = float("-inf")
    f_reeval_best = float("-inf")
    step = 0
    reevaluating_steps = 0
    iteration = 0
    evaluation = 0
    done = False
    initialize_result_file = True
    iterations_since_best_found = 0
    is_reevaluating_flag = False
    ESNOF_ARRAY_SIZE = 20000
    ESNOF_ref_objective_values =  np.full(ESNOF_ARRAY_SIZE, np.nan, dtype=np.float64)
    ESNOF_observed_objective_values = np.full(ESNOF_ARRAY_SIZE, np.nan, dtype=np.float64)
    ESNOF_index = 0
    ESNOF_stop = False

    result_file_path = None

    new_best_found = False

    SAVE_EVERY = 5
    mutex = Lock()


    def __init__(self, result_file_folder_path: str, params: Parameters, deletePreviousResults=False, limit_the_amount_of_written_lines=False):
        print("Perhaps if we change the project to: how can we set the inner length and quantity in an online manner? BC that is what we will ultimately need to do for the CONFLOT project...")
        self.params = params
        self.sw_reeval.pause()
        self.result_file_path = result_file_folder_path + f"/{params.get_result_file_name()}.txt"
        self.deletePreviousResults = deletePreviousResults
        self.max_frames = self.params.max_frames
        self.limit_the_amount_of_written_lines = limit_the_amount_of_written_lines 
        assert self.params.experiment_mode in ("reevaleachvsend", "proposedmethod","adaptstepspermorphology")
        if self.params.experiment_mode == "proposedmethod":
            self.ESNOF_reset_for_next_solution()


    def next_step(self):
        if self.is_reevaluating_flag:
            self.reevaluating_steps += 1
        else:
            self.step += 1


    def next_inner(self, f_partial=None):
        if self.is_reevaluating_flag:
            return


        def print_ESNOF(str): # for debug purposes
            return
            with open("../../esnof_log", "a") as f:
                print(str, file=f, flush=True)



        
        if self.params.experiment_mode == "proposedmethod":
            min_steps_per_morph = int(self.params.max_frames * self.params.esnof_min_time_ratio)
            max_steps_per_morph = int(self.params.max_frames * self.params.max_time_per_morph_ratio)
            steps_current_morph = self.step - self.ESNOF_steps_begining
            print_ESNOF(f"next_inner() -> {steps_current_morph} steps. ({min_steps_per_morph, max_steps_per_morph})")
            
            assert self.params.method_mode in ["standard","gesp","ctalatii","standard"]
            if self.ESNOF_index == 0:
                self.ESNOF_observed_objective_values[0] = f_partial
            else:
                # Force observe objective values to monotone increasing
                self.ESNOF_observed_objective_values[self.ESNOF_index] = max(f_partial, self.ESNOF_observed_objective_values[self.ESNOF_index-1])


            print_ESNOF("-")
            # with np.printoptions(threshold=np.inf):
            #     print(np.array([el for el in self.ESNOF_ref_objective_values if not np.isnan(el)]))
            #     print(np.array([el for el in self.ESNOF_observed_objective_values if not np.isnan(el)]))

            if "standard" in self.params.method_mode:
                pass
            else:
                if steps_current_morph < min_steps_per_morph:
                    self.last_grace_ESNOF_index = self.ESNOF_index
                    print_ESNOF("--continue_not_min_frames--")
                elif steps_current_morph >= max_steps_per_morph:
                    print_ESNOF("--stopp_max_frames_per_morph--")
                    self.ESNOF_stop = True
                else:
                    if "gesp" in self.params.method_mode:

                        current_index = self.ESNOF_index
                        prev_index = current_index - self.last_grace_ESNOF_index

                        print_ESNOF(f"gesp ({prev_index}, {current_index}) , ({self.ESNOF_ref_objective_values[prev_index]}, {self.ESNOF_observed_objective_values[current_index]})")

                        if self.ESNOF_ref_objective_values[prev_index] > self.ESNOF_observed_objective_values[current_index]:
                            print_ESNOF("--gesp-stop--")
                            self.ESNOF_stop = True
                        else:
                            print_ESNOF("--gesp-continue--")
                            pass
                    if "ctalatii" in self.params.method_mode:
                        current_index = self.ESNOF_index
                        prev_index = round(current_index * self.params.ctalatii_reference_ratio)

                        print_ESNOF(f"ctalatii ({prev_index}, {current_index}) , ({self.ESNOF_observed_objective_values[prev_index]}, {self.ESNOF_observed_objective_values[current_index]})")

                        if prev_index > 0 and self.ESNOF_observed_objective_values[current_index] == self.ESNOF_observed_objective_values[prev_index]:
                            print_ESNOF("--ctalatii-stop--")
                            self.ESNOF_stop = True
                        else:
                            print_ESNOF("--ctalatii-continue--")
                            pass
            print_ESNOF("-")
            self.ESNOF_index += 1


    def next_outer(self, f_observed, controller_size, controller_size2, morphology_size):
        if self.is_reevaluating_flag:
            return

        assert not f_observed is None
        self.f_observed = f_observed
        self.controller_size = controller_size
        self.morphology_size = morphology_size
        self.controller_size2 = controller_size2

        if self.step > self.max_frames:
            print("Finished at", self.max_frames,"frames.")
            exit(0)

        self.evaluation += 1
        if self.params.experiment_mode == "reevaleachvsend" or (self.params.experiment_mode == "proposedmethod" and not self.ESNOF_stop):
            self.check_if_best(level=2)
        self.ESNOF_reset_for_next_solution()
        if self.limit_the_amount_of_written_lines and self.step < 0.99 * self.max_frames and not self.is_reevaluating_flag:
            pass # we skip writing result.
        else:
            self.write_to_file(level=2)
        self.print_progress()


    def next_reeval(self, f_reeval_observed, controller_size, controller_size2, morphology_size):
        if self.params.experiment_mode == "proposedmethod":
            raise ValueError("ERROR: Experiment proposedmethod should have no reeval.")
        self.f_reeval_observed = f_reeval_observed
        self.controller_size = controller_size
        self.morphology_size = morphology_size
        self.controller_size2 = controller_size2
        self.check_if_best(level=3)
        self.write_to_file(level=3)
        self.is_reevaluating_flag = False
        self.sw_reeval.pause()
        self.sw.resume()

    def ESNOF_reset_for_next_solution(self):
        self.ESNOF_steps_begining = self.step
        self.ESNOF_index = 0
        self.ESNOF_stop = False
        self.ESNOF_observed_objective_values = np.full(self.ESNOF_ARRAY_SIZE, np.nan, dtype=np.float64)


    def ESNOF_load_new_references(self):
        np.copyto(self.ESNOF_ref_objective_values, self.ESNOF_observed_objective_values)
        self.ESNOF_observed_objective_values

    def check_if_best(self, level):
        # print("Checking for best found.")
        if level == 2:
            if self.f_observed > self.f_best:
                self.prev_f_best = self.f_best
                self.f_best = self.f_observed
                if self.params.experiment_mode == "proposedmethod":
                    self.ESNOF_load_new_references()
                if self.params.experiment_mode == "reevaleachvsend":
                    self.is_reevaluating_flag = True
                self.sw_reeval.resume()
                self.sw.pause()
                print("best_found! (level 2)")
        if level == 3:
            if self.f_reeval_observed > self.f_reeval_best:
                self.f_reeval_best = self.f_reeval_observed
                self.params.reevaluated_was_new_best_flags.append(1)
                if self.step + self.reevaluating_steps <= self.max_frames:
                    self.new_best_found = True
                print("best_found! (level 3)")
            else:
                self.params.reevaluated_was_new_best_flags.append(0)
                # What I previously believed -> # By reseting the best found, we can recreate reseting and not reseting the best found in the results. 
                # What I believe now -> If we do this, there are two problems even with reestructuring results.
                #   1) The best found reevaluated could be found thanks to a lower level to best_f, hence the saved animations don't represent reality
                #   2) Running with this is extremely slow, because a lot of reevaluations are made, specially with low parameters such as inner_length_proportion = 0.1 
                # print("Reseting to best_f to self.prev_f_best")
                # self.f_best = self.prev_f_best


    # @monitor_results
    def get_inner_length(self):
        if self.is_reevaluating_flag:
            return self.params._default_inner_length
        else:
            if self.params.experiment_mode == "reevaleachvsend":
                return int(self.params._inner_length_proportion * self.params._default_inner_length)
            elif self.params.experiment_mode == "proposedmethod":
                return self.params._default_inner_length
            else:
                raise ValueError("Experiment not recognized.")



    # @monitor_results
    def get_inner_quantity(self):
        if self.is_reevaluating_flag:
            return self.params._default_inner_quantity
        else:
            if self.params.experiment_mode == "reevaleachvsend":
                return int(self.params._inner_quantity_proportion * self.params._default_inner_quantity)
            elif self.params.experiment_mode == "proposedmethod":
                if self.params.method_mode == "standard":
                    return self.params._get_inner_quantity_absolute()
                else:
                    return 2000000000
            else:
                raise ValueError("Experiment not recognized.")


    def log_to_file(self, log_content):
        self.mutex.acquire()
        try:
            with open(self.result_file_path, "a") as f:
                f.write(log_content)
        finally:
            self.mutex.release()


    def check_result_file_and_write_header(self):
        if self.deletePreviousResults:
            if os.path.isfile(self.result_file_path):
                os.remove(self.result_file_path)
            if os.path.isfile(self.result_file_path):
                print(f"File {self.result_file_path} already exists and could not be deleted.")
                raise FileExistsError()
            self.deletePreviousResults = False
        else:
            if os.path.isfile(self.result_file_path):
                print(f"File {self.result_file_path} already exists. You can udse deletePreviousResults=True to delete previous results.")
                raise FileExistsError()
        with open(self.result_file_path, "a") as f:
            f.write("level,evaluation,f_best,f,controller_size,controller_size2,morphology_size,time,time_including_reeval,step,step_including_reeval\n")
        self.initialize_result_file = False

    def write_to_file(self, level):
        self.mutex.acquire()
        if self.initialize_result_file:
            try:
                assert os.path.isdir(os.path.dirname(os.path.abspath(self.result_file_path)))
                self.check_result_file_and_write_header()
            except AssertionError:
                print(f"ERROR: result directory {os.path.dirname(os.path.abspath(self.result_file_path))} does not exist. Exit...")
                exit(1)
            except FileExistsError:
                print(f"ERROR: result file {os.path.dirname(os.path.abspath(self.result_file_path))} already exists and was not deleted. Exit...")
                exit(1)

        with open(self.result_file_path, "a") as f:
            if level == 2:
                f.write(f"{level},{self.evaluation},{self.f_best},{self.f_observed},{self.controller_size},{self.controller_size2},{self.morphology_size},{self.sw.get_time()},{self.sw.get_time() + self.sw_reeval.get_time()},{self.step},{self.step + self.reevaluating_steps}\n")
            elif level == 3:
                f.write(f"{level},{self.evaluation},{self.f_reeval_best},{self.f_reeval_observed},{self.controller_size},{self.controller_size2},{self.morphology_size},{self.sw.get_time()},{self.sw.get_time() + self.sw_reeval.get_time()},{self.step},{self.step + self.reevaluating_steps}\n")
        self.mutex.release()


    def get_seed(self):
        return self.params.seed

    def get_video_label(self):
        return f"{self.params.experiment_mode}_{self.params.experiment_index}_{self.step}_{self.f_reeval_best}_{self.f_reeval_observed}"
    
    def print_progress(self):
        print_every = 60.0*20.0 # every 20 mins
        if self.sw_print_progress.get_time() > print_every:
            self.sw_print_progress.reset()
            print("Progress:", self.step / self.max_frames, ", left:", convert_from_seconds((self.sw.get_time() + self.sw_reeval.get_time()) / self.step * (self.max_frames - self.step)),", time:", datetime.now(), flush=True)


import os
import fcntl

import fcntl

class Lock:
    def __init__(self, filenames):
        if isinstance(filenames, str):
            self.filenames = [filenames]
        self.files = [open(filename, 'r+') for filename in self.filenames]

    def __enter__(self):
        for file in self.files:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        if len(self.filenames) == 1:
            return self.files[0]
        else:
            return self.files

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            file.close()





class experimentProgressTracker:

    def __init__(self, progress_filename, start_index, max_index):

        self.min_exp_time = 0.0
        self.progress_filename, self.start_index, self.max_index = progress_filename, start_index, max_index

        self.start_ref = time.time()
        self.last_ref = dict()
        self.done = False
        
        path = Path('./'+ progress_filename)
        if not path.is_file() or os.stat(path).st_size < 4:
            with open(progress_filename,"a") as f:
                print("idx,done", file=f)
        self._clean_unfinished_jobs_from_log()


    def _clean_unfinished_jobs_from_log(self):
        with Lock(self.progress_filename) as f:
            lines = f.readlines()
            f.seek(0)
            processed_lines = [line for line in lines if line.endswith(',1\n')]
            self.n_experiments_done_initially = len(processed_lines)
            f.writelines(["idx,done\n"]+processed_lines)
            f.truncate()

    def _get_next_index(self):
        with Lock(self.progress_filename) as f:
            content = f.read()
            for i in range(self.start_index, self.max_index+1):
                if f"{i}," not in content:
                    self.last_ref[i] = time.time()
                    print(f"{i},0", file=f, flush=True) # Mark experiment index in progress
                    return i
        return None

    def get_next_index(self):
        i = self._get_next_index()
        if i==None:
            self.done = True
            print("No more experiments left.")
            exit(0)

        print("------------\nWorking on experiment",i,"\n--------------")
        return i
    
    def mark_index_done(self, i):
        if self.done:
            exit(0)
        assert time.time() - self.last_ref[i] > self.min_exp_time

        with Lock(self.progress_filename) as f:
            lines = []
            lines = f.readlines()
            index = lines.index(f"{i},0\n")
            lines[index] = f"{i},1\n"
            f.seek(0)
            f.truncate()
            f.writelines(lines)
            n_experiments_done_total = len([0 for line in lines if line.endswith(',1\n')])
            n_experiments_done_this_session = n_experiments_done_total - self.n_experiments_done_initially
            n_experiments_left = self.max_index - n_experiments_done_total
            elapsed_time = time.time() - self.start_ref
            time_left = elapsed_time / n_experiments_done_this_session * n_experiments_left


            with open(self.progress_filename+"_log.txt","a") as f_log:
                f_log.write(f"{i},{n_experiments_left},{convert_from_seconds(time_left)}, {convert_from_seconds(elapsed_time)}\n")
