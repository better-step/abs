cmake_minimum_required(VERSION 3.22)
project(abs LANGUAGES CXX)

# ------------------------------------------------------------
# Dependencies (pulled once per wheel build)
# ------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cpp")

include(CPM)          # from cpp/CPM.cmake
include(pybind11)     # from cpp/pybind11.cmake
include(eigen)        # from cpp/eigen.cmake

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Build the pybind11 extension  (name = abspy)
# ------------------------------------------------------------
pybind11_add_module(abspy cpp/poisson_disk_downsample.cpp)
target_link_libraries(abspy PRIVATE pybind11::module Eigen3::Eigen)

# Install into the pure-python package dir that ends up inside the wheel
install(TARGETS abspy LIBRARY DESTINATION abs)
