

"""
Robotic controller definitions
"""

from .wave import WaveGenerator
from .abstract import Control

__all__ = [Control, WaveGenerator]


def load(toolbox, settings):
    ctrl = settings.get('control', 'type')
    if ctrl == 'wave':
        if 'sigma' in settings['control']:
            # We divide the sigma with the number of parameters which it will
            # be applied to ensuring that mutation will not move the parameters
            # further than expected
            ampl = settings.getfloat('control', 'sigma') / 4.0
            toolbox.register('control', WaveGenerator, ampl_sigma=ampl)
        else:
            ampl = settings.getfloat('control', 'amplitude_sigma')
            freq = settings.getfloat('control', 'frequency_sigma')
            phase = settings.getfloat('control', 'phase_sigma')
            offset = settings.getfloat('control', 'offset_sigma')
            toolbox.register('control', WaveGenerator, ampl_sigma=ampl,
                             freq_sigma=freq, phase_sigma=phase,
                             offset_sigma=offset)
    else:
        raise NotImplementedError("No controller named '{}' supported"
                                  .format(ctrl))
