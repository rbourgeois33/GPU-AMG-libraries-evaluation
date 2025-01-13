cmake_minimum_required(VERSION 3.10)

# Set the project name and specify C++ standard
project(GinkgoExample LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the Ginkgo library (adjust the path to the installed Ginkgo)
set(GINKGO_DIR "${CMAKE_SOURCE_DIR}/ginkgo/build/install")

# Add the Ginkgo library include and library paths
find_package(Ginkgo REQUIRED HINTS ${GINKGO_DIR})

# Add an executable target for the main file
add_executable(main main.cpp)

# Include the Ginkgo header files
target_include_directories(main PRIVATE ${GINKGO_DIR}/include)

# Include mynvtx header file directory
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/mynvtx)

# Link the Ginkgo library to the executable
target_link_libraries(main PRIVATE Ginkgo::ginkgo)