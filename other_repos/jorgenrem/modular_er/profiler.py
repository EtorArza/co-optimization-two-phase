

"""
Timing/profiling support for DEAP
"""
import time


class Profiler(object):
    """Profiler intended to be registered with 'toolbox'.

    Use the '__call__' method to start a trace which returns a 'Trace' object
    that is responsible for timing."""
    # List of profiles to write to CSV
    _traces = []

    def __call__(self, generation, name):
        trace = Trace()
        self._traces.append((generation, name, trace))
        return trace

    def traces(self):
        """Extract traces"""
        return [(gen, name, trace.duration)
                for gen, name, trace in self._traces
                if trace.duration is not None]


class Trace(object):
    """Trace object that is responsible for timing.

    Users should explicitly call 'stop' to get the most accurate timing"""
    def __init__(self):
        self._start = time.perf_counter()
        self.duration = None

    def stop(self):
        """Stop trace"""
        stop = time.perf_counter()
        self.duration = stop - self._start

    def __del__(self):
        self.stop()
