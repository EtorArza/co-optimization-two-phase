import time

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
    n_step = 0
    outer_iterations = 0
    result_file_path = None

    def __init__(self, result_file_path):
        self.result_file_path = result_file_path
        with open(self.result_file_path, "a") as f:
            f.write("f,steps,time,outer_iterations")
            
        self.reset()

    def reset(self):
        self.sw.reset()
        self.f_best = float("-inf")
        self.f_current = None
        self.n_step = 0
        self.outer_iterations = 0


    def next_step(self, current_f):
        self.n_step += 1
        self.f_current = current_f
        # print("next_step()", self, current_f)

    def next_inner(self, f_inner=None):
        if not f_inner is None:
            self.f_current = f_inner
        self.check_if_best(self.f_current)
        self.write_to_file()
        print("next_inner()", self, self.f_best)


    def next_outer(self):
        self.outer_steps += 1
        print("next_outer()", self, self.f_best)
        self.check_if_best(self.f_current)

    def check_if_best(self, f):
        print("Checking for best found.")
        if f > self.f_best:
            self.f_best = f
            print("best_found!")
    
    def write_to_file(self):
        with open(self.result_file_path, "a") as f:
            f.write(f"{self.f_best}, {self.n_step}, {self.sw.get_time()}, {self.outer_iterations}\n")
