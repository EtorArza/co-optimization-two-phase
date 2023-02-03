cmake_minimum_required(VERSION 3.0)

# set(Python_FIND_VIRTUALENV "ONLY" CACHE PATH "Use only virtual env.")
# set(Python_ROOT_DIR "/workspace/easybuild/x86_64/software/Python/3.9.5-GCCcore-10.3.0/include/python3.9/" CACHE PATH "Path to python")
# set(PYTHON_EXECUTABLE " ../../../venv/bin/python" CACHE PATH "Path to the venv python interpreter.")


project(RobotDesign)

# Use double precision in Bullet
add_compile_options(-DBT_USE_DOUBLE_PRECISION)

add_subdirectory(core)
add_subdirectory(examples/python_bindings)
add_subdirectory(examples/rule_viewer)

file(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
