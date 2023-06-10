import time
from threading import Thread, Lock
import numpy as np
from datetime import datetime
import os
import functools


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
            self.ESNOF_t_max = round(self._default_inner_quantity / 50)


        elif framework_name == "robogrammar":
            self.max_frames = 10240000 # max_frames=40960000 is the default value if we consider 5000 iterations as in the example in the GitHub.
            self.env_name_list = ["FlatTerrainTask"]
            self._default_inner_quantity = 64 # The number of random samples to generate when simulating each step.
            self._default_inner_length = 128 # Episode length.
            self.non_resumable_param = "quantity"
            self.ESNOF_t_max = self._default_inner_length


        elif framework_name == "tholiao":
            self.max_frames = 250000 # max_frames=2000000 is the default value (total steps)
            self.env_name_list = ["FlatTerrainTask"]
            self._default_inner_quantity =  50 # The number of controllers tested per morphology
            self._default_inner_length = 400 # Steps per episode
            self.non_resumable_param = "length"
            self.ESNOF_t_max = self._default_inner_length

        elif framework_name == "gymrem2d":
            # max_frames=3002177 is the default value on average. However, the framework 
            # itself is not a nested optimization approach. The framework by default evaluates 10000.
            # using 3002177 frames would not be enough because each morphology is now evaluated multiple times.
            # hence, we consider a budget x4 of the original, 

            self.max_frames = 12008708
            self.env_name_list = ["default"]
            self._default_inner_quantity =  128 # <- Need to do some parameter search!!!
            self._default_inner_length = 100 # Represents the speed of the early stopping blue wall in percentage
            self.non_resumable_param = "length"
            self.ESNOF_t_max = self._default_inner_length


        elif framework_name == "jorgenrem":
            self.max_frames = 480000000 # max_frames=480000000 is the default value
            self.env_name_list = ["default"]
            self._default_inner_quantity =  10 # <- Need to do some parameter search!!!
            self._default_inner_length = 4800 # Represents the number of steps per controller tested
            self.non_resumable_param = "length"
            self.ESNOF_t_max = self._default_inner_length

        else:
            raise ValueError(f"Framework {framework_name} not found.")


        params = self._get_parameter_list()[experiment_index]

        if "reevaleachvsend" in params:
            self.seed, self._inner_quantity_proportion, self._inner_length_proportion, self.env_name, self.experiment_mode = params
        elif "incrementalandesnof" in params:
            self.seed, self.minimum_non_resumable_param, self.time_grace, self.env_name, self.experiment_mode = params
            self.ESNOF_t_grace = round(self.time_grace * self.ESNOF_t_max)
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


    def _get_parameter_list_old(self):
        nseeds_old = 20
        import itertools
        res = []
        seed_list = list(range(2,2 + nseeds_old))

        # reevaleachvsend
        _inner_quantity_proportion_list = [0.25, 0.5, 0.75, 1.0]
        _inner_length_proportion_list =   [0.25, 0.5, 0.75, 1.0]
        experiment_mode_list = ["reevaleachvsend"]
        params_with_undesired_combinations = list(itertools.product(_inner_quantity_proportion_list, _inner_length_proportion_list, self.env_name_list, experiment_mode_list, seed_list))
        params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item or item[0] == item[1]] # remove the combinations containining 2 different parameters != 1.0.
        res += params_with_undesired_combinations


        # # incrementalandesnof
        # minimum_non_resumable_param_list = [0.2, 1.0]
        # time_grace_list = [0.2, 1.0]
        # experiment_mode_list = ["incrementalandesnof"]
        # params_with_undesired_combinations = list(itertools.product(seed_list, minimum_non_resumable_param_list, time_grace_list, self.env_name_list, experiment_mode_list))
        # params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item or item[1] == item[2]] # remove the combinations containining 2 different parameters != 1.0.
        # res += params_with_undesired_combinations

        # adaptstepspermorphology
        target_probability = [0.75]
        _start_quantity_proportion = [0.1]
        experiment_mode_list = ["adaptstepspermorphology"]
        params_with_undesired_combinations = list(itertools.product(seed_list, target_probability, _start_quantity_proportion, self.env_name_list, experiment_mode_list))
        res += params_with_undesired_combinations

        return res


    def _get_parameter_list(self):
        import itertools
        res = []
        seed_list = list(range(1,1 + self.nseeds))

        # reevaleachvsend
        _inner_quantity_proportion_list = [0.1, 0.25, 0.5, 0.75, 1.0]
        _inner_length_proportion_list =   [0.1, 0.25, 0.5, 0.75, 1.0]
        experiment_mode_list = ["reevaleachvsend"]
        params_with_undesired_combinations = list(itertools.product(seed_list, _inner_quantity_proportion_list, _inner_length_proportion_list, self.env_name_list, experiment_mode_list))
        params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item[1:3]] # remove the combinations in which one of the parameters is not 1.0.
        res += params_with_undesired_combinations


        # # incrementalandesnof
        # minimum_non_resumable_param_list = [0.2, 1.0]
        # time_grace_list = [0.2, 1.0]
        # experiment_mode_list = ["incrementalandesnof"]
        # params_with_undesired_combinations = list(itertools.product(minimum_non_resumable_param_list, time_grace_list, self.env_name_list, experiment_mode_list, seed_list))
        # params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item or item[0] == item[1]] # remove the combinations containining 2 different parameters != 1.0.
        # res += params_with_undesired_combinations

        # # adaptstepspermorphology
        # target_probability = [0.75]
        # _start_quantity_proportion = [0.1]
        # experiment_mode_list = ["adaptstepspermorphology"]
        # params_with_undesired_combinations = list(itertools.product(target_probability, _start_quantity_proportion, self.env_name_list, experiment_mode_list, seed_list))
        # res += params_with_undesired_combinations

        return res


    def reindex_all_result_files(self, directory, extension):
        old_params = self._get_parameter_list_old()
        params = self._get_parameter_list()

        all_result_file_paths: list[str] = []
        for filepath in os.listdir(directory):
            f = os.path.join(directory, filepath)
            if os.path.isfile(f) and extension in f:
                all_result_file_paths.append(f)
 
        new_index_exists = np.zeros(10000)
        largest_new_index = 0
        for i, el in enumerate(old_params):
            old_index = i

            if el not in params:
                for filepath in all_result_file_paths:
                    if f"_{old_index}_" in filepath:
                        os.rename(filepath, filepath.replace(f"_{old_index}_", f"_legacyfile{old_index}_"))
                        continue
                continue

            new_index = params.index(el)
            largest_new_index = max(new_index, largest_new_index)
            for filepath in all_result_file_paths:
                if f"_{old_index}_" in filepath:
                    os.rename(filepath, filepath.replace(f"_{old_index}_", f"_placeholdertext{new_index}_"))
                    new_index_exists[new_index] = 1
                    print(el,old_index, "->",new_index)

        for filepath in os.listdir(directory):
            f = os.path.join(directory, filepath)
            if os.path.isfile(f) and extension in f:
                os.rename(f, f.replace("placeholdertext", ""))                

        print("Empty result indexes:")
        previous = 1
        for i in range(largest_new_index):

            if new_index_exists[i] == previous: # if there is no change
                continue

            elif new_index_exists[i] == 0:
                first_index_0 = i
                previous = 0
            
            elif new_index_exists[i] == 1:
                print(first_index_0, "-", i-1)
                previous = 1


        print("done!")
        print("Run \n\nfind results | grep legacyfile | xargs rm\n\n to remove old experiment files.")


    def get_result_file_name(self):
        if self.experiment_mode == "reevaleachvsend":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self._inner_quantity_proportion}_{self._inner_length_proportion}_{self.seed}"
        if self.experiment_mode == "incrementalandesnof":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self.minimum_non_resumable_param}_{self.time_grace}_{self.seed}"
        if self.experiment_mode == "adaptstepspermorphology":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self.target_probability}_{self._start_quantity_proportion}_{self.seed}"
        else:
            raise NotImplementedError(f"Result file name not implemented for {self.experiment_mode} yet.")

    def get_n_experiments(self):
        return len(self._get_parameter_list())

    def print_parameters(self):
        print("Total number of executions:", self.get_n_experiments())
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
    incrementalandesnof

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
    ESNOF_ARRAY_SIZE = 2000
    ESNOF_ref_objective_values =  np.full(ESNOF_ARRAY_SIZE, np.nan, dtype=np.float64)
    ESNOF_observed_objective_values = np.full(ESNOF_ARRAY_SIZE, np.nan, dtype=np.float64)
    ESNOF_index = 0
    ESNOF_stop = False

    result_file_path = None

    new_best_found = False

    SAVE_EVERY = 5
    mutex = Lock()


    def __init__(self, result_file_folder_path: str, params: Parameters, deletePreviousResults=False):
        print("Perhaps if we change the project to: how can we set the inner length and quantity in an online manner? BC that is what we will ultimately need to do for the CONFLOT project...")
        self.params = params
        self.sw_reeval.pause()
        self.result_file_path = result_file_folder_path + f"/{params.get_result_file_name()}.txt"
        self.deletePreviousResults = deletePreviousResults
        self.max_frames = params.max_frames
        assert params.experiment_mode in ("reevaleachvsend", "incrementalandesnof","adaptstepspermorphology")


    def print_to_result_file(self, msg_string):
        self.mutex.acquire()
        try:
            with open(self.result_file_path, "a") as f:
                    f.write(msg_string)
        finally:
            self.mutex.release()

    def next_step(self):
        if self.is_reevaluating_flag:
            self.reevaluating_steps += 1
        else:
            self.step += 1


    def next_inner(self, f_partial=None):
        if self.is_reevaluating_flag:
            return
        # print("next_inner() -> ", self.step, "steps.")
        if self.params.experiment_mode == "incrementalandesnof":
            self.ESNOF_observed_objective_values[self.ESNOF_index] = f_partial
            # print("-")
            # with np.printoptions(threshold=np.inf):
            #     print(np.array([el for el in self.ESNOF_ref_objective_values if not np.isnan(el)]))
            #     print(np.array([el for el in self.ESNOF_observed_objective_values if not np.isnan(el)]))
            if self.ESNOF_index > self.params.ESNOF_t_grace and self.ESNOF_ref_objective_values[self.ESNOF_index - self.params.ESNOF_t_grace] > f_partial:
                # print("--stop--")
                self.ESNOF_stop = True
            else:
                # print("--continue--")
                pass
            # print("-")
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
        if self.params.experiment_mode in ("reevaleachvsend", "adaptstepspermorphology") or not self.ESNOF_stop:
            self.check_if_best(level=2)
        self.ESNOF_reset_for_next_solution()
        self.write_to_file(level=2)
        self.print_progress()


    def next_reeval(self, f_reeval_observed, controller_size, controller_size2, morphology_size):
        if self.params.experiment_mode == "incrementalandesnof":
            raise ValueError("ERROR: Experiment incrementalandesnof should have no reeval.")
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
                if self.params.experiment_mode == "incrementalandesnof":
                    self.ESNOF_load_new_references()
                if self.params.experiment_mode in ("reevaleachvsend","adaptstepspermorphology"):
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

            # Set current inner_quantity
            if self.params.experiment_mode == "adaptstepspermorphology":
                if len(self.params.reevaluated_was_new_best_flags) > 3:
                    current_proportion = np.mean(self.params.reevaluated_was_new_best_flags)
                    self.params._inner_quantity_proportion += 0.1 * np.sign(self.params.target_probability - current_proportion)
                    self.params._inner_quantity_proportion = max(0.1, self.params._inner_quantity_proportion)
                    self.params._inner_quantity_proportion = min(1.0, self.params._inner_quantity_proportion)
                    print("New inner quantity proportion: ",self.params._inner_quantity_proportion)


    def get_inner_non_resumable_increasing(self):

        param_proportion = self.params.minimum_non_resumable_param  + (1.0 - self.params.minimum_non_resumable_param) * (self.step / self.max_frames)
        if self.params.non_resumable_param == "length":
            res = round(param_proportion * self.params._default_inner_length)
        if self.params.non_resumable_param == "quantity":
            res = round(param_proportion * self.params._default_inner_quantity)

        return res

    # @monitor_results
    def get_inner_length(self):
        if self.is_reevaluating_flag:
            return self.params._default_inner_length
        else:
            if self.params.experiment_mode == "reevaleachvsend":
                return int(self.params._inner_length_proportion * self.params._default_inner_length)
            elif self.params.experiment_mode == "adaptstepspermorphology":
                return self.params._default_inner_length
            elif self.params.experiment_mode == "incrementalesnof":
                if self.params.non_resumable_param == "length":
                    return self.get_inner_non_resumable_increasing()
                if self.params.non_resumable_param == "quantity":
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
            elif self.params.experiment_mode == "adaptstepspermorphology":
                return self.params._get_inner_quantity_absolute()
            elif self.params.experiment_mode == "incrementalesnof":
                if self.params.non_resumable_param == "quantity":
                    return self.get_inner_non_resumable_increasing()
                if self.params.non_resumable_param == "length":
                    return self.params._default_inner_quantity
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
class lock(object):
    def __init__(self, file_name):
        self.lock_file_name = file_name + ".lock"
        self.start_time = time.time()

    def __enter__(self):
        timeout = 10  # Maximum timeout in seconds

        while True:
            if not os.path.exists(self.lock_file_name):
                try:
                    with open(self.lock_file_name, 'x') as f:
                        print(f"Lock adquired by thread {threading.get_ident()}.",  flush=True)
                        return
                except FileExistsError:
                    continue

            elif time.time() - self.start_time >= timeout:
                raise TimeoutError(f"Thread {threading.get_ident()} failed to acquire lock within the timeout period.")

            else:

                print(f"Thread {threading.get_ident()} wating {time.time() - self.start_time}...") 
                time.sleep(0.1)  # Sleep for a short duration before retrying

    def __exit__(self, exception_type, exception_value, traceback):
        timeout = 10  # Maximum timeout in seconds
        exit_time = time.time()
        time.sleep(0.5)  # Sleep for a short duration before trying to delete lock file.
        if not os.path.exists(self.lock_file_name):
            print(f"File seems to not exist for thread {threading.get_ident()} on exit.", flush=True)
            exit(1)
        else:
            print(f"Thread {threading.get_ident()} deleting lock file...", flush=True)
            while os.path.exists(self.lock_file_name): 
                os.remove(self.lock_file_name)
                time.sleep(0.1)
                if time.time() - exit_time >= timeout:
                    raise TimeoutError(f"Thread {threading.get_ident()} failed to delete lock file.")
            print(f"Deleted lock. Lock lasted {time.time() -  self.start_time} s", flush=True)
                



        





class experimentProgressTracker:

    def __init__(self, progress_filename, start_index, max_index):

        from pathlib import Path
        import pandas as pd
        import time
        self.progress_filename, self.start_index, self.max_index = progress_filename, start_index, max_index
        self.n_experiments_done_this_session = 0

        self.start_ref = time.time()
        self.last_ref = dict()
        self.done = False
        
        path = Path('./'+ progress_filename)
        if not path.is_file():
            with open(progress_filename,"a") as f:
                print("idx", file=f)

        df = pd.read_csv(progress_filename)
        self.done_idx_list = []
        if df.shape[0] > 0:
            self.done_idx_list = list(df["idx"])


        self.done_idx_list += [i for i in range(start_index) if i not in self.done_idx_list]
        self.done_idx_list.sort()

        print(self.done_idx_list)
    
    def get_next_index(self):
        for i in range(self.max_index):
            if not i in self.done_idx_list:
                self.last_ref[i] = time.time()
                self.done_idx_list.append(i)
                print("------------\nWorking on experiment",i,"\n--------------")
                return i
        self.done = True
        print("No more experiments left.")
        return None
    
    def mark_index_done(self, i):
        assert time.time() - self.last_ref[i] > 7.0
        self.n_experiments_done_this_session += 1
        n_experiments_left = self.max_index - len(self.done_idx_list)
        elapsed_time = time.time() - self.start_ref
        time_left = elapsed_time / self.n_experiments_done_this_session * n_experiments_left

        with open(self.progress_filename+"_log.txt","a") as f:
            f.write(f"{i},{n_experiments_left},{convert_from_seconds(time_left)}, {convert_from_seconds(elapsed_time)}\n")
        with open(self.progress_filename,"a") as f:
            print(i, file=f)
