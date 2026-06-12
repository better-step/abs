################################################################################
# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.14.0")
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

project(abs LANGUAGES CXX)
################################################################################

OPTION(ABS_BUILD_BINARY "Build tests" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cpp/")


# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Polyfem library
################################################################################

# dependencies
if(NOT ABS_BUILD_BINARY)
    find_package(pybind11 CONFIG REQUIRED)
endif()
include(eigen)
include(nanospline)


################################################################################
# Subdirectories
################################################################################
if(ABS_BUILD_BINARY)
    add_executable(abspy_bin cpp/poisson_disk_downsample.cpp)
    target_compile_features(abspy_bin PRIVATE cxx_std_17)
    target_compile_definitions(abspy_bin PRIVATE -DABS_BUILD_BINARY)
    target_link_libraries(abspy_bin PRIVATE Eigen3::Eigen nanospline::nanospline)
else()
    pybind11_add_module(abspy cpp/poisson_disk_downsample.cpp)
    target_link_libraries(abspy PRIVATE pybind11::module Eigen3::Eigen nanospline::nanospline)
    #install(TARGETS abspy DESTINATION .)
    install(TARGETS abspy LIBRARY DESTINATION .)
endif()






