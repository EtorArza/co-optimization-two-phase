#!/usr/bin/env python

"""
Simple wave generator controller

The innards of this controller is identical to a CPG controller except that
this does not take into account neighboring controllers.
"""
from .abstract import Control
import numpy as np

TWO_PI = np.pi * 2.


def normalize(value, min_range, max_range):
    """Normalize value based on ranges"""
    assert 0.0 < value < 1.0, "Can't normalize with value outside (0, 1)"
    value_range = max_range - min_range
    return value * value_range


class WaveGenerator(Control):
    """Simple wave generation control"""

    def __init__(self, ampl_sigma=0.1, freq_sigma=None, phase_sigma=None,
                 offset_sigma=None):
        super().__init__()
        # Select random values for internal values
        self.amplitude = np.random.uniform(-1.57, 1.57)
        self.frequency = np.random.uniform(0.2, 2.)
        self.phase = np.random.uniform(-1., 1.)
        self.offset = np.random.uniform(-1.57, 1.57)
        # Setup mutation configuration
        self._ampl_sigma = ampl_sigma
        self._freq_sigma = freq_sigma if freq_sigma else ampl_sigma
        self._phase_sigma = phase_sigma if phase_sigma else ampl_sigma
        self._offset_sigma = offset_sigma if offset_sigma else ampl_sigma
        self.normalize()

    def __call__(self, _, time):
        output = (self.amplitude * np.sin(time * self.frequency
                                          + self.phase * TWO_PI)
                  + self.offset)
        return min(max(output, -1.57), 1.57)

    def normalize(self):
        """Normalize sigma values"""
        self._ampl_sigma = normalize(self._ampl_sigma, -1.57, 1.57)
        self._freq_sigma = normalize(self._freq_sigma, 0.2, 2.)
        self._phase_sigma = normalize(self._phase_sigma, -1., 1.)
        self._offset_sigma = normalize(self._offset_sigma, -1.57, 1.57)

    def mutate(self):
        # First mutate parameters
        amplitude = np.random.normal(self.amplitude, self._ampl_sigma)
        freq = np.random.normal(self.frequency, self._freq_sigma)
        phase = np.random.normal(self.phase, self._phase_sigma)
        offset = np.random.normal(self.offset, self._offset_sigma)
        # Then limit to correct range
        self.amplitude = WaveGenerator.limit(amplitude, -1.57, 1.57)
        self.frequency = WaveGenerator.limit(freq, 0.2, 2.)
        self.phase = WaveGenerator.limit(phase, -1., 1.)
        self.offset = WaveGenerator.limit(offset, -1.57, 1.57)

    @staticmethod
    def limit(value, low, high):
        """Limit the value between [low, high] through a bounce back effect"""
        if value < low:
            step = low - value
            value = low + step
        elif value > high:
            step = value - high
            value = high - step
        return np.clip(value, low, high)

    def __repr__(self):
        return ("WaveGenerator(amplitude: {:.2f}, frequency: {:.2f}, phase: {:.2f}, offset: {:.2f})"
                .format(self.amplitude, self.frequency, self.phase,
                        self.offset))
