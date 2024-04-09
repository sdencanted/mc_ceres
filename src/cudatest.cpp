#include "mc_gradient_bilinear.h"
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

// #include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv){
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    std::vector<float> x={1};
    std::vector<float> y={1};
    std::vector<float> t={1};
    McGradientBilinear *mc_gr = new McGradientBilinear(0, 0,0, 0, x, y, t, 1,1,1);
    ceres::GradientProblem problem(mc_gr);
    // cudaSetDeviceFlags(cudaDeviceScheduleYield);
    // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    float* image_,*image_del_theta_x_,*image_del_theta_y_,*image_del_theta_z_;
    double residuals[3],gradient[3];
    int height_=180,width_=240;
    
    cudaMalloc(&image_, height_*width_ * sizeof(float));
    cudaMalloc(&image_del_theta_x_, height_*width_ * sizeof(float));
    cudaMalloc(&image_del_theta_y_, height_*width_* sizeof(float));
    cudaMalloc(&image_del_theta_z_, height_*width_ * sizeof(float));


    float *contrast_block_sum;
    float *contrast_del_x_block_sum;
    float *contrast_del_y_block_sum;
    float *contrast_del_z_block_sum;
    float *means;
    float *contrast_block_sum_cpu;
    int gridSize=85;//with a block size of 512 for a 180*240 image

    checkCudaErrors(cudaMalloc((void **)&contrast_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMalloc(&means, 4 * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&contrast_block_sum_cpu, sizeof(float) * 4));


    float image_two[height_*width_];
    std::fill_n(image_two,height_*width_,2);
    image_two[0]=0;


    
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
        getContrastDelBatchReduce(image_, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_, residuals, gradient, height_, width_,contrast_block_sum,contrast_del_x_block_sum,contrast_del_y_block_sum,contrast_del_z_block_sum,means,contrast_block_sum_cpu);
        nvtxRangePop(); // Ends NVTX range
        std::cout<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<" "<<residuals[0]<<std::endl;
        
        uint8_t output_image[height_*width_];
        // cv::Mat mat(height_, width_, CV_8U, output_image);
    }
    cudaFree(image_);
    cudaFree(image_del_theta_x_);
    cudaFree(image_del_theta_y_);
    cudaFree(image_del_theta_z_);
    
    checkCudaErrors(cudaFree(contrast_block_sum));
    checkCudaErrors(cudaFree(contrast_del_x_block_sum));
    checkCudaErrors(cudaFree(contrast_del_y_block_sum));
    checkCudaErrors(cudaFree(contrast_del_z_block_sum));
    checkCudaErrors(cudaFreeHost(contrast_block_sum_cpu));
    checkCudaErrors(cudaFree(means));
}