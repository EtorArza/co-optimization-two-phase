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


class NestedOptimization:

    sw = stopwatch()
    f_best = float("-inf")
    f_current = None
    steps = 0
    iterations = 0
    evaluations = 0
    write_header = True
    is_first_step_this_inner = True
    _last_saved_sw = stopwatch()
    SAVE_EVERY = 1.0
    mutex = Lock()


    def __init__(self, result_file_path, mode):
        self.result_file_path = result_file_path
        self.mode = mode
        assert mode in ("save_all", "standard")
        self.reset()

    def reset(self):
        self.sw.reset()
        self._last_saved_sw.reset()
        self.f_best = float("-inf")
        self.f_current = None
        self.steps = 0
        self.iterations = 0
        self.evaluations = 0
        is_first_step_this_inner = True


    def next_step(self, current_f):
        self.steps += 1
        self.f_current = current_f
        if self.mode == "save_all":
            if self._last_saved_sw.get_time() > self.SAVE_EVERY or self.is_first_step_this_inner:
                self.write_to_file()
                self._last_saved_sw.reset()
                self.is_first_step_this_inner = False
        print("next_step()", self, self.steps, current_f)

    def next_inner(self, f_inner=None):
        self.iterations += 1
        if not f_inner is None:
            self.check_if_best(f_inner)
        if self._last_saved_sw.get_time() > self.SAVE_EVERY or self.mode == "save_all":
            self.write_to_file()
            self._last_saved_sw.reset()
        self.is_first_step_this_inner = True
        print("next_inner()", self, self.f_best)


    def next_outer(self):
        self.evaluations += 1
        self.is_first_step_this_inner = True
        print("next_outer()", self, self.f_best)
        self.check_if_best(self.f_current)

    def check_if_best(self, f):
        print("Checking for best found.")
        if f > self.f_best:
            self.f_best = f
            print("best_found!")
    
    def write_to_file(self):
        self.mutex.acquire()
        try:
            with open(self.result_file_path, "a") as f:
                if self.write_header:
                    f.write("f,time,steps,iterations,evaluations\n")
                    self.write_header = False
                f.write(f"{self.f_best}, {self.sw.get_time()}, {self.steps}, {self.iterations}, {self.evaluations}\n")
        finally:
            self.mutex.release()

