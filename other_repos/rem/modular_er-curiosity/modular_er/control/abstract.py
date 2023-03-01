#!/usr/bin/env python

"""
Abstract controller to illustrate all necessary functions that should be
supported by other controllers
"""


class Control(object):
    """Abstract controller class to ensure subclasses have all methods
    available"""

    def __call__(self, observation, time):
        """Function called to produce new output"""
        raise NotImplementedError("Abstract class")

    def reset(self, module):
        """Reset internal state before beginning a new simulation

        The 'module' parameter is the morphological module this instance is
        connected to. The idea is that classes that need to connect nodes can
        traverse the tree and set it up here without storing a reference to the
        morphology it self.

        This function is called after the environment is reset and is a chance
        for the controller to update according to the correct morphology as it
        appears in the simulation"""
        pass

    def mutate(self):
        """Mutate control parameters"""
        raise NotImplementedError("Abstract class")
