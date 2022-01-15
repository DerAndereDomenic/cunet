cmake_minimum_required(VERSION "3.8")

project(CUNET LANGUAGES CXX C CUDA VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

file (GLOB_RECURSE cunet_files LIST_DIRECTORIES false "lib/**.c*"
                                                      "lib/**.h*")

add_library(CUNET STATIC "${cunet_files}")

add_executable(Example "src/source.cu" "src/TestModel.h")

target_link_libraries(Example CUNET)

include_directories("lib/include"
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_definitions(CUNET PUBLIC _USE_MATH_DEFINES)
target_compile_definitions(Example PUBLIC _USE_MATH_DEFINES)

set_target_properties(Example PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

#string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")