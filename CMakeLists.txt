################################################################################
# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14.0")
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

project(abs LANGUAGES CXX)
################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cpp/")


# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Polyfem library
################################################################################

# dependencies
include(pybind11)
include(eigen)


################################################################################
# Subdirectories
################################################################################
pybind11_add_module(abspy cpp/poisson_disk_downsample.cpp)
target_link_libraries(abspy PRIVATE pybind11::module Eigen3::Eigen)
#install(TARGETS abspy DESTINATION .)
install(TARGETS abspy LIBRARY DESTINATION abs)






