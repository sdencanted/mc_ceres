cmake_minimum_required(VERSION 3.5)

project(cm_ceres)

find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse SparseLinearAlgebraLibrary LAPACK SchurSpecializations)

# helloworld
add_executable(main src/main.cpp)
target_link_libraries(main Ceres::ceres)