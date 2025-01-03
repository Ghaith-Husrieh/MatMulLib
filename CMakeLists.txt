cmake_minimum_required(VERSION 3.30.2)
project(MatMulLib VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB LIB_SOURCES ${PROJECT_SOURCE_DIR}/src/lib/*.c)
add_library(MatMulLib SHARED ${LIB_SOURCES})

set_target_properties(MatMulLib PROPERTIES 
    PREFIX ""
)

# Note: Executable for testing purposes during development
add_executable(MatMulLibTest ${PROJECT_SOURCE_DIR}/src/test.c)
target_link_libraries(MatMulLibTest PRIVATE MatMulLib)