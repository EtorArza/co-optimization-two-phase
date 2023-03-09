import time
from threading import Thread, Lock
import numpy as np
from datetime import datetime

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

    nseeds = 20

    def __init__(self, framework_name: str, experiment_index: int):

        self.experiment_index = experiment_index
        self.framework_name = framework_name

        if framework_name == "evogym":

            # If default_train_iters < 100 (to test stuff) it does not work. This is because the performance
            # of the model is only saved every 50 iterations, and if the parameter inner 
            # inner_quantity_proportion is 0.5, we get 50 iterations when default_train_iters = 100.

            self.max_frames = 8633000 # Default 34532000 considering 250 morphologies evaluated (easy tasks).
            self.env_name_list = ["Walker-v0"]
            self.default_inner_quantity = 1000 # The number of the iterations of the PPO algorithm.
            self.default_inner_length = 128 # Episode length.
            self.non_resumable_param = "length"
            self.ESNOF_t_max = round(self.default_inner_quantity / 50)


        elif framework_name == "robogrammar":
            self.max_frames = 10240000 # max_frames=40960000 is the default value if we consider 5000 iterations as in the example in the GitHub.
            self.env_name_list = ["FlatTerrainTask"]
            self.default_inner_quantity = 64 # The number of random samples to generate when simulating each step.
            self.default_inner_length = 128 # Episode length.
            self.non_resumable_param = "quantity"
            self.ESNOF_t_max = self.default_inner_length

        else:
            raise ValueError(f"Framework {framework_name} not found.")


        params = self._get_parameter_list()[experiment_index]

        if "reevaleachvsend" in params:
            self.inner_quantity_proportion, self.inner_length_proportion, self.env_name, self.experiment_mode, self.seed = params
        elif "incrementalandesnof" in params:
            self.minimum_non_resumable_param, self.time_grace, self.env_name, self.experiment_mode, self.seed = params
            self.ESNOF_t_grace = round(self.time_grace * self.ESNOF_t_max)
        else:
            raise ValueError(f"params variable {params} does not contain a recognized experiment")


    def get_inner_quantity_absolute(self):
        return int(self.inner_quantity_proportion * self.default_inner_quantity)

    def get_inner_length_absolute(self):
        return int(self.inner_length_proportion * self.default_inner_length)


    def _get_parameter_list(self):
        import itertools
        res = []
        seed_list = list(range(2,2 + self.nseeds))

        # reevaleachvsend
        inner_quantity_proportion_list = [0.25, 0.5, 0.75, 1.0]
        inner_length_proportion_list =   [0.25, 0.5, 0.75, 1.0]
        experiment_mode_list = ["reevaleachvsend"]
        params_with_undesired_combinations = list(itertools.product(inner_quantity_proportion_list, inner_length_proportion_list, self.env_name_list, experiment_mode_list, seed_list))
        params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item or item[0] == item[1]] # remove the combinations containining 2 different parameters != 1.0.
        res += params_with_undesired_combinations



        # incrementalandesnof
        minimum_non_resumable_param_list = [0.2, 1.0]
        time_grace_list = [0.2, 1.0]
        experiment_mode_list = ["incrementalandesnof"]
        params_with_undesired_combinations = list(itertools.product(minimum_non_resumable_param_list, time_grace_list, self.env_name_list, experiment_mode_list, seed_list))
        params_with_undesired_combinations = [item for item in params_with_undesired_combinations if 1.0 in item or item[0] == item[1]] # remove the combinations containining 2 different parameters != 1.0.
        res += params_with_undesired_combinations

        return res

    def get_result_file_name(self):
        if self.experiment_mode == "reevaleachvsend":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self.inner_quantity_proportion}_{self.inner_length_proportion}_{self.seed}"
        if self.experiment_mode == "incrementalandesnof":
            return f"{self.experiment_mode}_{self.experiment_index}_{self.env_name}_{self.minimum_non_resumable_param}_{self.time_grace}_{self.seed}"
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
    write_header = True
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


    def __init__(self, result_file_folder_path: str, params: Parameters):
        print("Perhaps if we change the project to: how can we set the inner length and quantity in an online manner? BC that is what we will ultimately need to do for the CONFLOT project...")
        self.params = params
        self.sw_reeval.pause()
        self.result_file_path = result_file_folder_path + f"/{params.get_result_file_name()}.txt"
        self.max_frames = params.max_frames
        assert params.experiment_mode in ("reevaleachvsend", "incrementalandesnof")


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
        if self.params.experiment_mode == "reevaleachvsend" or not self.ESNOF_stop:
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
                self.f_best = self.f_observed
                self.new_best_found = True
                if self.params.experiment_mode == "incrementalandesnof":
                    self.ESNOF_load_new_references()
                if self.params.experiment_mode == "reevaleachvsend":
                    self.is_reevaluating_flag = True
                self.sw_reeval.resume()
                self.sw.pause()
                print("best_found! (level 2)")
        if level == 3:
            if self.f_reeval_observed > self.f_reeval_best:
                self.f_reeval_best = self.f_reeval_observed
                if self.step + self.reevaluating_steps <= self.max_frames:
                    self.new_best_found = True
                print("best_found! (level 3)")



    def get_inner_non_resumable_increasing(self):

        param_proportion = self.params.minimum_non_resumable_param  + (1.0 - self.params.minimum_non_resumable_param) * (self.step / self.max_frames)
        if self.params.non_resumable_param == "length":
            res = round(param_proportion * self.params.default_inner_length)
        if self.params.non_resumable_param == "quantity":
            res = round(param_proportion * self.params.default_inner_quantity)

        return res



    def write_to_file(self, level):
        self.mutex.acquire()
        try:
            with open(self.result_file_path, "a") as f:
                if self.write_header:
                    f.write("level,evaluation,f_best,f,controller_size,controller_size2,morphology_size,time,time_including_reeval,step,step_including_reeval\n")
                    self.write_header = False
                if level == 2:
                    f.write(f"{level},{self.evaluation},{self.f_best},{self.f_observed},{self.controller_size},{self.controller_size2},{self.morphology_size},{self.sw.get_time()},{self.sw.get_time() + self.sw_reeval.get_time()},{self.step},{self.step + self.reevaluating_steps}\n")
                elif level == 3:
                    f.write(f"{level},{self.evaluation},{self.f_reeval_best},{self.f_reeval_observed},{self.controller_size},{self.controller_size2},{self.morphology_size},{self.sw.get_time()},{self.sw.get_time() + self.sw_reeval.get_time()},{self.step},{self.step + self.reevaluating_steps}\n")
        finally:
            self.mutex.release()

    def get_seed(self):
        return 2

    def get_video_label(self):
        return f"{self.params.experiment_mode}_{self.params.experiment_index}_{self.step}_{self.f_reeval_best}_{self.f_reeval_observed}"
    
    def print_progress(self):
        print_every = 60.0*20.0 # every 20 mins
        if self.sw_print_progress.get_time() > print_every:
            self.sw_print_progress.reset()
            print("Progress:", self.step / self.max_frames, ", left:", convert_from_seconds((self.sw.get_time() + self.sw_reeval.get_time()) / self.step * (self.max_frames - self.step)),", time:", datetime.now(), flush=True)



