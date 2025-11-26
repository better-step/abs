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

# # dependencies
# find_package(pybind11 CONFIG REQUIRED)
# include(eigen)


# ################################################################################
# # Subdirectories
# ################################################################################
# pybind11_add_module(abspy cpp/poisson_disk_downsample.cpp)
# target_link_libraries(abspy PRIVATE pybind11::module Eigen3::Eigen)
# #install(TARGETS abspy DESTINATION .)
# install(TARGETS abspy LIBRARY DESTINATION .)









################################################################################
# HDF5 test
################################################################################


# h5pp + HDF5 library
add_executable(hdf5_reader cpp/hdf5_reader.cpp)
include(hdf5)
target_link_libraries(hdf5_reader PUBLIC hdf5::hdf5)

include(h5pp)
target_link_libraries(hdf5_reader PUBLIC h5pp::h5pp)


include(nanospline)
target_link_libraries(hdf5_reader PUBLIC nanospline::nanospline)





