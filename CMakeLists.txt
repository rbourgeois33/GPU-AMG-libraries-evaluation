cmake_minimum_required(VERSION 3.10)

# Set the project name and specify C++ standard
project(AMG_EVAL LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(lib/amgcl)

add_executable(amgcl_hip_eval amgcl_hip_eval.cpp)

target_link_libraries(amgcl_hip_eval
    PRIVATE
        amgcl::amgcl
)
