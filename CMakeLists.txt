cmake_minimum_required(VERSION 3.10)

# Set the project name and specify C++ standard
project(GinkgoExample LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Point CMake to AMGX headers
include_directories("${CMAKE_SOURCE_DIR}/lib/AMGX/include")

# Point CMake to AMGX libraries
link_directories("${CMAKE_SOURCE_DIR}/lib/AMGX/build")

# Create the executable from your source file
add_executable(AMGX_eval AMGX_eval.cpp)

# Link against the AMGX shared library (AMGXsh).
# If your library name differs, adjust accordingly (e.g., AMGX, libAMGXsh, etc.).
target_link_libraries(AMGX_eval amgxsh)