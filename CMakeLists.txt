cmake_minimum_required(VERSION 3.30.2)
project(MatMulLib VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")

set(LIB_SOURCES
    ${PROJECT_SOURCE_DIR}/src/lib/tensor.c
    ${PROJECT_SOURCE_DIR}/src/lib/matmul.c
    ${PROJECT_SOURCE_DIR}/src/lib/random.c
)
add_library(MatMulLib SHARED ${LIB_SOURCES})

# Note: Executable for testing purposes during development
add_executable(MatMulLibTest ${PROJECT_SOURCE_DIR}/src/test.c)
target_link_libraries(MatMulLibTest PRIVATE MatMulLib)