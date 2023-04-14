

"""
Genome/Phenotype individual encoding module
"""

from .direct import load as direct_load


ENCODING_MAPPING = {'direct': direct_load}


def load(toolbox, config):
    """Load encoding"""
    enc_type = config['encoding']['type']
    if enc_type in ENCODING_MAPPING:
        ENCODING_MAPPING[enc_type](toolbox, config)
    else:
        raise NotImplementedError("Encoding: '{}' not supported, available: {}"
                                  .format(enc_type, ENCODING_MAPPING.keys()))
