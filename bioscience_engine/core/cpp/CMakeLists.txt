cmake_minimum_required(VERSION 3.15)
project(BioScienceEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)

include_directories(bioscience_engine/core/cpp/include)

file(GLOB_RECURSE SOURCES "bioscience_engine/core/cpp/src/*.cpp")

add_library(bioscience_core STATIC ${SOURCES})
target_link_libraries(bioscience_core 
    Eigen3::Eigen 
    OpenMP::OpenMP_CXX
    tiff
)

target_compile_options(bioscience_core PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

# Optional: Build standalone executable for testing
add_executable(test_core tests/cpp/test_main.cpp)
target_link_libraries(test_core bioscience_core)