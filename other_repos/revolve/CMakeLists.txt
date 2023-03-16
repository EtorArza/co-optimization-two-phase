cmake_minimum_required(VERSION 2.8.12)

include_directories(thirdparty/nlopt/install_dir/usr/local/include)
project(revolve)

if (BUILD_RASPBERRY)
    add_subdirectory(thirdparty/PIGPIO)
endif()

add_subdirectory(cpprevolve)