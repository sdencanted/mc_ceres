cmake_minimum_required(VERSION 3.5)

# set( ENV{BLA_VENDOR} "ATLASpootis" )
# CUDA
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 87)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo") # enable line info
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_87 -gencode  arch=compute_87,code=sm_87)


project(cm_ceres
    VERSION 0.1
    LANGUAGES C CXX CUDA)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3")




# Option to enable/disable NVTX markers for improved profiling
set(USE_NVTX ON)

# add_compile_definitions(EIGEN_STACK_ALLOCATION_LIMIT=0)
find_package(CUDA 11.4 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
include_directories("${CUDA_INCLUDE_DIRS}")

# # Define link libraries for the target
# if(USE_NVTX)
#     message(STATUS "Finding NVTX")
#     # Find the nvtx library using custom cmake module
#     find_package(NVTX)
#     # If found, apply 
#     if(NVTX_FOUND)
#         message(STATUS "Find NVTX include at ${NVTX_INCLUDE_DIRS}")
#         set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
#         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
#         set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} "${NVTX_INCLUDE_DIRS}")
#         if(NVTX_VERSION VERSION_LESS "3")
#             set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} ${NVTX_LIBRARIES})
#         endif()
#     else()
#         # If not found, disable.
#         message(STATUS "NVTX not available")
#         SET(USE_NVTX "OFF" PARENT_SCOPE)    
#     endif()
# endif()
# find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse SparseLinearAlgebraLibrary LAPACK SchurSpecializations)
find_package(Ceres REQUIRED COMPONENTS EigenSparse)


# jetson-utils
find_package(jetson-utils REQUIRED)
include_directories(/usr/lib/aarch64-linux-gnu/tegra)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

# OpenCV
find_package (OpenCV 4.5.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Find opencv include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find opencv libraries: ${OpenCV_LIBRARIES}")

# GSL
find_package(GSL REQUIRED)

FIND_PACKAGE(dv-processing 1.7.0 REQUIRED)



include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# main
cuda_add_executable(main src/main.cpp src/reduce.cu src/motion_compensation.cu )
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 87)
# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(main
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    ${OpenCV_LIBS}
    nvtx3-cpp
    dv::processing
)

# main_txt
cuda_add_executable(main_txt src/main_txt.cpp src/reduce.cu src/motion_compensation.cu )
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 87)
# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main_txt
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(main_txt
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    ${OpenCV_LIBS}
    nvtx3-cpp
    dv::processing
)

# aedat_mc
cuda_add_executable(aedat_mc src/aedat_mc.cpp src/reduce.cu src/motion_compensation.cu )
set_property(TARGET aedat_mc PROPERTY CUDA_ARCHITECTURES 87)
# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(aedat_mc
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(aedat_mc
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    ${OpenCV_LIBS}
    nvtx3-cpp
    dv::processing
)
# main gsl
cuda_add_executable(main_gsl src/main_gsl.cpp src/reduce.cu src/motion_compensation.cu )
# set_target_properties(main_gsl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
    nvtx3-cpp
)
# plot
cuda_add_executable(plot src/plot.cpp src/reduce.cu src/motion_compensation.cu )
# set_target_properties(plot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
add_subdirectory(utils/NVTX/c)

# cudatest
cuda_add_executable(cudatest src/cudatest.cpp src/motion_compensation.cu )
set_property(TARGET cudatest PROPERTY CUDA_ARCHITECTURES 87)
# set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cudatest
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cudatest 
    Ceres::ceres
    ${CUDA_LIBRARIES}
    jetson-utils
    nvtx3-cpp
)