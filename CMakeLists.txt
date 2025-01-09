cmake_minimum_required(VERSION 3.30.2)
project(MatMulLib VERSION 0.1.0 LANGUAGES C)

# Set C Standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)


# Include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set library prefixes
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")

# Source files for the library
set(LIB_SOURCES
    ${PROJECT_SOURCE_DIR}/src/lib/tensor.c
    ${PROJECT_SOURCE_DIR}/src/lib/matmul.c
    ${PROJECT_SOURCE_DIR}/src/lib/random.c
)

# Add the shared library
add_library(MatMulLib SHARED ${LIB_SOURCES})

# Find and enable OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_C_FOUND)
    target_link_libraries(MatMulLib PRIVATE OpenMP::OpenMP_C)
endif()

# Add an executable for testing during development
add_executable(MatMulLibTest ${PROJECT_SOURCE_DIR}/src/test.c)
target_link_libraries(MatMulLibTest PRIVATE MatMulLib)