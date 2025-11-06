cmake_minimum_required(VERSION 3.18)
project(GLOPBaseline CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Fetch OR-Tools (now that path has no spaces, this should work)
message(STATUS "Fetching OR-Tools")
include(FetchContent)
FetchContent_Declare(
    ortools
    GIT_REPOSITORY https://github.com/google/or-tools.git
    GIT_TAG v9.11
    GIT_SHALLOW TRUE
)
set(BUILD_DEPS ON CACHE BOOL "" FORCE)
set(BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ortools)

# Eigen3 dependency
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitLab...")
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen)
endif()

# Library target
add_library(glop_baseline_lib
    glop_solver.cpp
)
target_include_directories(glop_baseline_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(glop_baseline_lib PUBLIC ortools::ortools Eigen3::Eigen)

# Google Test for testing
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Test executable
add_executable(glop_test test_glop_solver.cpp)
target_link_libraries(glop_test
    glop_baseline_lib
    GTest::gtest
    GTest::gtest_main
)

# Benchmark executable
add_executable(glop_benchmark benchmark.cpp)
target_link_libraries(glop_benchmark glop_baseline_lib)

# Main example executable
add_executable(glop_main main.cpp)
target_link_libraries(glop_main glop_baseline_lib)
