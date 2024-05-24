#include "mc_gradient_bilinear.h"
#include "motion_compensation_float.h"
// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_float.h"

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
    
    cudaMallocHost(&image_, height_*width_ * sizeof(float));
    cudaMallocHost(&image_del_theta_x_, height_*width_ * sizeof(float));
    cudaMallocHost(&image_del_theta_y_, height_*width_* sizeof(float));
    cudaMallocHost(&image_del_theta_z_, height_*width_ * sizeof(float));


    float *contrast_block_sum;
    float *contrast_del_x_block_sum;
    float *contrast_del_y_block_sum;
    float *contrast_del_z_block_sum;
    float *means;
    float *contrast_block_sum_cpu;
    int gridSize=85;//with a block size of 512 for a 180*240 image

    checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&contrast_del_x_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&contrast_del_y_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void **)&contrast_del_z_block_sum, gridSize * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&means, 4 * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&contrast_block_sum_cpu, sizeof(float) * 4));


    float image_two[height_*width_];
    std::fill_n(image_two,height_*width_,2);
    image_two[0]=0;


    
    std::cout<<image_two[0]<<std::endl;


    using namespace std::chrono;
    
    uint64_t ms = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();    
    
    for(int i=0;i<1;i++){
        nvtx3::scoped_range r{"batch reduce"};
        std::fill_n(residuals,3,0);
        std::fill_n(gradient,3,0);
        // std::fill_n(image_,height_*width_,1);
        // std::fill_n(image_del_theta_x_,height_*width_,2);
        // std::fill_n(image_del_theta_y_,height_*width_,3);
        // std::fill_n(image_del_theta_z_,height_*width_,4);
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
        // for(int u=0; u<height_*width_;u++){
        //     image_[u]=u/10;
        //     image_del_theta_x_[u]=u/20;
        //     image_del_theta_y_[u]=u*3/10;
        //     image_del_theta_z_[u]=u/40;
        // }
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
        std::cout<<"reduction means "<<means[0]<<" "<<means[1]<<" "<<means[2]<<" "<<means[3]<<std::endl;
        std::fill_n(means,4,0);
        for(int i=0;i<height_*width_;i++){
            means[0]+=image_[i];
            means[1]+=image_del_theta_x_[i];
            means[2]+=image_del_theta_y_[i];
            means[3]+=image_del_theta_z_[i];
        }

        std::cout<<"CPU means "<<means[0]/(height_*width_)<<" "<<means[1]/(height_*width_)<<" "<<means[2]/(height_*width_)<<" "<<means[3]/(height_*width_)<<std::endl;
        // for(int i=0;i<128;i++){
        //     std::cout<<i<<" "<<contrast_block_sum[i]<<" "<<contrast_del_x_block_sum[i]<<" "<<contrast_del_y_block_sum[i]<<" "<<contrast_del_z_block_sum[i]<<std::endl;
        // }
        uint8_t output_image[height_*width_];
        // cv::Mat mat(height_, width_, CV_8U, output_image);
    }
    cudaFreeHost(image_);
    cudaFreeHost(image_del_theta_x_);
    cudaFreeHost(image_del_theta_y_);
    cudaFreeHost(image_del_theta_z_);
    
    checkCudaErrors(cudaFreeHost(contrast_block_sum));
    checkCudaErrors(cudaFreeHost(contrast_del_x_block_sum));
    checkCudaErrors(cudaFreeHost(contrast_del_y_block_sum));
    checkCudaErrors(cudaFreeHost(contrast_del_z_block_sum));
    checkCudaErrors(cudaFreeHost(contrast_block_sum_cpu));
    checkCudaErrors(cudaFreeHost(means));
}