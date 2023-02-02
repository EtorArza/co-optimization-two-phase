import os
from setuptools import setup

# Change working directory to the one this file is in
os.chdir(os.path.dirname(os.path.realpath(__file__)))

inst_libs = ['py_value', 'py_sim', 'py_robot_design', 'py_robot', 'py_render', 'py_optim', 'py_graph', 'py_prop', 'py_eigen_geometry']

for lib in inst_libs:
    setup(
    name=lib,
    version='0.1.0',
    packages=[lib],
    package_dir={lib: '..'}
    )