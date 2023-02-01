import os
import re
import sys
import platform
import subprocess

# cmake /workspace/scratch/jobs/earza/fakeHome/other_repos/evogym/evogym/simulator -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/workspace/scratch/jobs/earza/fakeHome/other_repos/evogym/build/lib.linux-x86_64-3.9/evogym -DPYTHON_EXECUTABLE=/workspace/easybuild/x86_64/software/Python/3.9.5-GCCcore-10.3.0/bin/python -DGLEW_INCLUDE_DIR=/workspace/scratch/jobs/earza/fakeHome/glew-2.1.0/include/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=glew-2.1.0/


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(os.path.dirname(os.path.abspath(__file__))) # /workspace/scratch/jobs/earza/fakeHome/other_repos/evogym

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DGLEW_INCLUDE_DIR='+os.path.dirname(os.path.abspath(__file__))+'/../../glew-2.1.0/include/',
                      '-DCMAKE_PREFIX_PATH='+os.path.dirname(os.path.abspath(__file__))+'/../../glew-2.1.0/',
                      '-DCMAKE_BUILD_TYPE=Release',
                      '-DPYTHON_EXECUTABLE=' + sys.executable]


        cfg = 'Debug' if self.debug else 'Release'
        cfg = 'Release'
        # cfg = 'Debug'
        build_args = ['--config', cfg]
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name="evogym",
    packages=['evogym', 'evogym.envs'],
    package_dir={
        'evogym': 'evogym',
        'evogym.envs': 'evogym/envs'},
    package_data={
        "evogym.envs": [os.path.join('sim_files', '*.json')] #["*.json", "*.sob"],
    },
    ext_modules=[CMakeExtension('evogym.simulator_cpp', 'evogym/simulator')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
