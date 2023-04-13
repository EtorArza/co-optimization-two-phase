#!/usr/bin/env python

"""
Evolutionary Algorithm definitions
"""

from .single import run as single_run
from .map_elites import run as map_run
from .nsga2 import run as nsga2_run


EA = {'single': single_run, 'map-elites': map_run, 'nsga2': nsga2_run}


def load(config):
    """Load the desired EA based on values in settings"""
    ea_type = config.get('experiment', 'ea')
    if ea_type in EA:
        return EA[ea_type]
    raise NotImplementedError("EA: '{}' not supported, supported: {}"
                              .format(ea_type, EA.keys()))
