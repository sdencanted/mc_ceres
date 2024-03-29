cmake_minimum_required(VERSION 3.5)

# CUDA
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_87,code=sm_87)
project(cm_ceres
    VERSION 0.1
    LANGUAGES C CXX CUDA)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3")

find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse SparseLinearAlgebraLibrary LAPACK SchurSpecializations)

# add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
find_package(CUDA 11.4 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
include_directories("${CUDA_INCLUDE_DIRS}")

# jetson-utils
find_package(jetson-utils REQUIRED)
include_directories(/usr/lib/aarch64-linux-gnu/tegra)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# OpenCV
find_package (OpenCV 4.0.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GSL
find_package(GSL REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# main
cuda_add_executable(main src/main.cpp src/reduce.cu src/motion_compensation.cu )
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(main
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    ${OpenCV_LIBS}
    GSL::gsl 
    GSL::gslcblas
)

# main gsl
cuda_add_executable(main_gsl src/main_gsl.cpp src/reduce.cu src/motion_compensation.cu )
set_target_properties(main_gsl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main_gsl
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(main_gsl
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    ${OpenCV_LIBS}
    GSL::gsl 
    GSL::gslcblas
)
# plot
cuda_add_executable(plot src/plot.cpp src/reduce.cu src/motion_compensation.cu )
set_target_properties(plot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(plot
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(plot
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    ${OpenCV_LIBS}
    GSL::gsl 
    GSL::gslcblas
)