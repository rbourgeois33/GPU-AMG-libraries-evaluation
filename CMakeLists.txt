cmake_minimum_required(VERSION 3.10)

# Set the project name and specify C++ standard
project(AMG_EVAL LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Ginkgo

# Add the Ginkgo library (adjust the path to the installed Ginkgo)
set(GINKGO_DIR "${CMAKE_SOURCE_DIR}/lib/ginkgo/build/install")

# Add the Ginkgo library include and library paths
find_package(Ginkgo REQUIRED HINTS ${GINKGO_DIR})

# Add an executable target for the ginkgo_eval file
add_executable(ginkgo_eval ginkgo_eval.cpp)

# Include the Ginkgo header files
target_include_directories(ginkgo_eval PRIVATE ${GINKGO_DIR}/include)

# Include mynvtx header file directory
target_include_directories(ginkgo_eval PRIVATE ${CMAKE_SOURCE_DIR}/lib/mynvtx)

# Link the Ginkgo library to the executable
target_link_libraries(ginkgo_eval PRIVATE Ginkgo::ginkgo)

# ---- AMGX

# Point CMake to AMGX headers
include_directories("${CMAKE_SOURCE_DIR}/lib/AMGX/include")

# Point CMake to AMGX libraries
link_directories("${CMAKE_SOURCE_DIR}/lib/AMGX/build")

# Create the executable from your source file
add_executable(AMGX_eval AMGX_eval.cpp)

# Link against the AMGX shared library (AMGXsh).
# If your library name differs, adjust accordingly (e.g., AMGX, libAMGXsh, etc.).
target_link_libraries(AMGX_eval amgxsh)

# # ---- amgcl-cuda

# Tell CMake that we need CUDA language:
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# E.g. for Volta (sm_70) or Turing (sm_75) or Ampere (sm_80) etc.
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)  # or just "75", etc.

add_subdirectory(lib/amgcl)

set_source_files_properties(
    amgcl_cuda_eval.cpp
    PROPERTIES
        LANGUAGE CUDA
)

add_executable(amgcl_cuda_eval amgcl_cuda_eval.cpp)

target_link_libraries(amgcl_cuda_eval
    PRIVATE
        amgcl::amgcl
        CUDA::cusparse
 )


# ---- hypre


# Specify the correct path to the Hypre library
set(HYPRE_DIR "${CMAKE_SOURCE_DIR}/lib/hypre/build/install") # Use the 'build' folder where the library is found

# Add the executable target for your file
add_executable(hypre_eval hypre_eval.cpp)

# Include the Hypre headers
target_include_directories(hypre_eval PRIVATE ${HYPRE_DIR}/include)

# Link the Hypre library
target_link_directories(hypre_eval PRIVATE ${HYPRE_DIR}/lib)
target_link_libraries(hypre_eval PRIVATE ${HYPRE_DIR}/lib/libHYPRE.a) # Static linking
