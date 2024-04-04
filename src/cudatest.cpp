#include "motion_compensation.h"
// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation.h"
#include "reduce.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <nvtx3/nvtx3.hpp>
#include <chrono>


int main(int argc, char **argv){
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    // cudaSetDeviceFlags(cudaDeviceScheduleYield);
    // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    float* image_,*image_del_theta_x_,*image_del_theta_y_,*image_del_theta_z_;
    double residuals[3],gradient[3];
    int height_=180,width_=240;
    cudaMalloc(&image_, height_*width_ * sizeof(float));
    cudaMalloc(&image_del_theta_x_, height_*width_ * sizeof(float));
    cudaMalloc(&image_del_theta_y_, height_*width_* sizeof(float));
    cudaMalloc(&image_del_theta_z_, height_*width_ * sizeof(float));
    float image_two[height_*width_];
    std::fill_n(image_two,height_*width_,2);
    image_two[0]=0;


    int cub_temp_size= getCubSize(image_, height_, width_);
    float* temp_storage;
    
    checkCudaErrors(cudaMalloc(&temp_storage, cub_temp_size));
    std::cout<<image_two[0]<<std::endl;


    using namespace std::chrono;
    
    uint64_t ms = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();    
    for(int i=0;i<100;i++){
        nvtx3::scoped_range r{"batch reduce"};
        std::fill_n(residuals,3,0);
        std::fill_n(gradient,3,0);
        // cudaMemcpy(image_,image_two,height_*width_*sizeof(float),cudaMemcpyDefault);
        // cudaMemcpy(image_del_theta_x_,image_two,height_*width_*sizeof(float),cudaMemcpyDefault);
        // cudaMemcpy(image_del_theta_y_,image_two,height_*width_*sizeof(float),cudaMemcpyDefault);
        // cudaMemcpy(image_del_theta_z_,image_two,height_*width_*sizeof(float),cudaMemcpyDefault);    
        // std::cout<<time(NULL)<<std::endl;
        // std::cout<<time(NULL)<<std::endl;
        // std::cout<<time(NULL)<<std::endl;
        // std::cout<<time(NULL)<<std::endl;
        // std::cout<<time(NULL)<<std::endl;
        // std::cout<<time(NULL)<<std::endl;
        one_step_kernel(ms,image_,height_*width_);
        ms++;
        one_step_kernel(ms,image_del_theta_x_,height_*width_);
        ms++;
        one_step_kernel(ms,image_del_theta_y_,height_*width_); 
        ms++;
        one_step_kernel(ms,image_del_theta_z_,height_*width_);
        ms++;

        nvtxRangePushA("my_function"); // Begins NVTX range
        getContrastDelBatchReduce(image_, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_, residuals, gradient, height_, width_,cub_temp_size, temp_storage);
        nvtxRangePop(); // Ends NVTX range
        std::cout<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<" "<<residuals[0]<<std::endl;
    }
    cudaFree(temp_storage);
    cudaFree(image_);
    cudaFree(image_del_theta_x_);
    cudaFree(image_del_theta_y_);
    cudaFree(image_del_theta_z_);
}