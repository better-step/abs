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
# library
################################################################################

# dependencies
find_package(pybind11 CONFIG REQUIRED)
include(eigen)
include(hdf5)
include(h5pp)
include(nanospline)

# ################################################################################
# # Subdirectories
# ################################################################################
pybind11_add_module(abspy cpp/poisson_disk_downsample.cpp cpp/hdf5_reader.cpp)
target_link_libraries(abspy PRIVATE pybind11::module Eigen3::Eigen hdf5::hdf5 h5pp::h5pp nanospline::nanospline)
target_include_directories(abspy PRIVATE cpp)
#install(TARGETS abspy DESTINATION .)
install(TARGETS abspy LIBRARY DESTINATION .)









################################################################################
# HDF5 test
################################################################################


# h5pp + HDF5 library






