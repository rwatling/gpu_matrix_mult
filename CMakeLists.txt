cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(gpu-mat-mult LANGUAGES CXX CUDA)

##### Variables and flags
set(GPU_LIB "_cuda")

set(MAIN_PROJECT "main")

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "60")
endif()

##### FILES
file(GLOB main "${CMAKE_SOURCE_DIR}/src/main.cu")

file(GLOB gpu_source_files
    "${CMAKE_SOURCE_DIR}/src/mat_update.cu")

##### Include directories
include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${GPU_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

##### GPU Library
add_library(${GPU_LIB} STATIC ${gpu_source_files})

# Target compile feautes
target_compile_features(${GPU_LIB} PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in ${GPU_LIB}
# to be built with -dc
set_target_properties(${GPU_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

##### Executables - MAIN
add_executable(${MAIN_PROJECT} ${main})

##### Link libraries - MAIN PROJECT
target_link_libraries(${MAIN_PROJECT} PRIVATE ${GPU_LIB})
#target_link_libraries(${MAIN_PROJECT} PRIVATE ${CPU_LIB})
target_link_libraries(${MAIN_PROJECT} PUBLIC -lnvidia-ml)
set_property(TARGET ${MAIN_PROJECT} PROPERTY CXX_STANDARD 14)