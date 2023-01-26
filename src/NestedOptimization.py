import time
from threading import Thread, Lock
import numpy as np

class stopwatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_t = time.time()
        self.pause_t=0

    def pause(self):
        self.pause_start = time.time()
        self.paused=True

    def resume(self):
        if self.paused:
            self.pause_t += time.time() - self.pause_start
            self.paused = False

    def get_time(self):
        return time.time() - self.start_t - self.pause_t

    def get_time_string_short_format(self):
        return "{:.4f}".format(self.get_time())

class NestedOptimization:

    sw = stopwatch()
    f_observed = float("-inf")
    f_best = float("-inf")
    step = 0
    iteration = 0
    evaluation = 0
    done = False
    write_header = True
    iterations_since_best_found = 0
    result_file_path = None
    mode = None
    need_reevaluate = False


    SAVE_EVERY = 5
    mutex = Lock()


    def __init__(self, result_file_path, mode, max_frames, inners_per_outer, inner_length_proportion):
        self.result_file_path = result_file_path
        self.mode = mode
        self.max_frames = max_frames
        self.inners_per_outer = inners_per_outer
        self.inner_length_proportion = inner_length_proportion
        assert mode in ("saveall", "standard")




    def next_step(self, f_observed):
        self.step += 1
        self.f_observed = f_observed
        if self.mode == "saveall":
            print("next_step()", self, self.step, f_observed)
            self.write_to_file(level=0)



    def next_inner(self, f_observed):
        self.iteration += 1
        self.f_observed = f_observed
        if self.mode == "saveall":
            self.write_to_file(level=1)
            print("next_inner()", self, self.f_best)


    def next_outer(self, f_observed):
        self.evaluation += 1
        self.f_observed = f_observed
        if not f_observed is None:
            self.check_if_best(f_observed)
        
        self.write_to_file(level=2)
        print("next_outer()", self, f_observed)
        if self.max_frames >= self.step:
            self.done = True

    def next_saverealobjective(self, real_f):
        self.f_observed = real_f
        self.need_reevaluate = False
        self.write_to_file(level=3)
        print("next_saverealobjective()", self, real_f)


    def check_if_best(self, f):
        # print("Checking for best found.")
        if f > self.f_best:
            self.f_best = f
            self.need_reevaluate = True
            print("best_found!")
    
    def write_to_file(self, level):
        self.mutex.acquire()
        try:
            with open(self.result_file_path, "a") as f:
                if self.write_header:
                    f.write("level,f_best,f,time,step,iteration,evaluation\n")
                    self.write_header = False
                f.write(f"{level},"+str(self.f_best if level != 3 else "nan")+ ", "+ str(self.f_observed if not self.f_observed is None else "nan") + f",{self.sw.get_time_string_short_format()},{self.step},{self.iteration},{self.evaluation}\n")
        finally:
            self.mutex.release()

