#!/usr/bin/env python
from setuptools import setup

setup(name='modular_er',
      description="Modular Evolutionary Robotics experiments",
      version='0.1.0',
      author="Jørgen Nordmoen and Frank Veenstra",
      install_requires=['gym_rem>=0.1', 'numpy>=1.17', 'deap>=1.3',
                        'tqdm>=4.36', 'termcolor>=1.1'],
      extras_require={
          'mpi': ["mpi4py>=3.0"]
      })
