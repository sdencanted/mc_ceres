#ifndef MC_GRADIENT_BILINEAR_H
#define MC_GRADIENT_BILINEAR_H
#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
// #include "glog/logging.h"

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
#include <pthread.h>
#include <sys/resource.h>
#include <omp.h>
// A CostFunction implementing motion compensation then calculating contrast, as well as the jacobian.
class McGradientBilinear final : public ceres::FirstOrderFunction
{

public:
    ~McGradientBilinear()
    {
        cudaFree(x_unprojected_);
        cudaFree(y_unprojected_);
        cudaFreeHost(x_prime_);
        cudaFreeHost(y_prime_);
        cudaFreeHost(bilinear_values_);
        cudaFree(t_);
        cudaFreeHost(image_);
        cudaFreeHost(image_del_theta_x_);
        cudaFreeHost(image_del_theta_y_);
        cudaFreeHost(image_del_theta_z_);

        checkCudaErrors(cudaFree(contrast_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_x_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_y_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_z_block_sum_));
        checkCudaErrors(cudaFreeHost(contrast_block_sum_cpu_));
        checkCudaErrors(cudaFree(means_));
    }
    McGradientBilinear(const float fx, const float fy, const float cx, const float cy,
                       std::vector<float> &x, std::vector<float> &y, std::vector<float> &t, const int height, const int width, const int num_events, int gridSize = 85) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width), num_events_(num_events)
    {
        // create pinned memory for x,y,t,image,image dels
        cudaMallocHost(&x_unprojected_, num_events_ * sizeof(float));
        cudaMallocHost(&y_unprojected_, num_events_ * sizeof(float));
        cudaMallocHost(&x_prime_, num_events_ * sizeof(float));
        cudaMallocHost(&y_prime_, num_events_ * sizeof(float));
        cudaMallocHost(&bilinear_values_, num_events_ * sizeof(float) * 16);
        cudaMallocHost(&t_, num_events_ * sizeof(float));
        cudaMallocHost(&image_, (height_) * (width_) * sizeof(float));
        cudaMallocHost(&image_del_theta_x_, (height_) * (width_) * sizeof(float));
        cudaMallocHost(&image_del_theta_y_, (height_) * (width_) * sizeof(float));
        cudaMallocHost(&image_del_theta_z_, (height_) * (width_) * sizeof(float));

        cudaMemsetAsync(image_, 0, (height_) * (width_) * sizeof(float));
        cudaMemsetAsync(image_del_theta_x_, 0, (height_) * (width_) * sizeof(float));
        cudaMemsetAsync(image_del_theta_y_, 0, (height_) * (width_) * sizeof(float));
        cudaMemsetAsync(image_del_theta_z_, 0, (height_) * (width_) * sizeof(float));
        checkCudaErrors(cudaMalloc((void **)&contrast_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&contrast_block_sum_cpu_, sizeof(float) * 4));

        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // float scale=t[num_events-1]-t[0];
        float scale = 1e6;
        float t_cpu[num_events_];
        // find the middle t
        float middle_t = (t[num_events_ - 1] + t[0]) / 2;

        for (int i = 1; i < num_events_; i++)
        {
            t_cpu[i] = (t[i] - middle_t) / scale;
        }
        cudaMemcpy(t_, t_cpu, num_events_ * sizeof(float), cudaMemcpyHostToDevice);

        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        float x_unprojected_cpu[num_events_];
        float y_unprojected_cpu[num_events_];
        for (int i = 0; i < num_events_; i++)
        {
            x_unprojected_cpu[i] = (x[i] - cx) / fx;
            y_unprojected_cpu[i] = (y[i] - cy) / fy;
        }
        cudaMemcpy(x_unprojected_, x_unprojected_cpu, num_events_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y_unprojected_, y_unprojected_cpu, num_events_ * sizeof(float), cudaMemcpyHostToDevice);
        cub_temp_size_ = getCubSize(image_, height_, width_);
        cudaMalloc(&temp_storage_, cub_temp_size_ * sizeof(float));
    }
    void tryCudaAllocMapped(float **ptr, size_t size, std::string ptr_name)
    {
        std::cout << "allocating cuda mem for " << ptr_name << std::endl;
        if (!cudaAllocMapped(ptr, size))
        {
            std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
        }
    }
    void ReplaceData(std::vector<float> &x, std::vector<float> &y, std::vector<float> &t, const int num_events)
    {
        num_events_ = num_events;

        cudaFree(x_unprojected_);
        cudaFree(y_unprojected_);
        cudaFreeHost(x_prime_);
        cudaFreeHost(y_prime_);
        cudaFreeHost(bilinear_values_);
        cudaFree(t_);

        cudaMalloc(&x_unprojected_, num_events_ * sizeof(float));
        cudaMalloc(&y_unprojected_, num_events_ * sizeof(float));
        cudaMallocHost(&x_prime_, num_events_ * sizeof(float));
        cudaMallocHost(&y_prime_, num_events_ * sizeof(float));
        cudaMallocHost(&bilinear_values_, num_events_ * sizeof(float) * 16);
        cudaMalloc(&t_, num_events_ * sizeof(float));
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const override
    {
        // pthread_setschedprio(pthread_self(),-10000);
        // setpriority(PRIO_PROCESS, pthread_self(), -10);
        nvtx3::scoped_range r{"Evaluate"};
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        bool do_jacobian = gradient != nullptr;
        // std::cout<<do_jacobian<<std::endl;
        // Populate image

        // fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], do_jacobian, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_);
        motionCompensateBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, parameters[0], parameters[1], parameters[2], bilinear_values_);
        // fillImageBilinearIntrinsics(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], do_jacobian, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_);
        cudaDeviceSynchronize();

        //  queue a thread to calculate mean while we accumulate the image
        meanBilinear(bilinear_values_, num_events_, means_, contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, height_, width_);
// Potentially OMP
// definitely ARM vector add

        nvtxRangePushA("OMP"); // Begins NVTX range
#pragma omp parallel num_threads(4) // Orin NX has 8 threads, use 4 for 4 different images
        {
            int image_id = omp_get_thread_num();
            int offset = image_id * num_events_ * 4;
            float *image_target = NULL;
            switch (image_id)
            {
            case 0:
                image_target = image_;
                break;
            case 1:
                image_target = image_del_theta_x_;
                break;
            case 2:
                image_target = image_del_theta_y_;
                break;
            case 3:
                image_target = image_del_theta_z_;
                break;
            }
            for (int i = 0; i < num_events_; i++)
            {
                int x = x_prime_[i];
                int y = y_prime_[i];
                // if (x >= 1 && x <= width_ - 2 && y >= 1 && y <= height_ - 2)
                if(x_prime_[i]>0)
                {
                    image_target[x+width_]++;
                // //     int idx4 = x +width_* y;
                // //     int idx3 = idx4 - 1;
                // //     int idx1 = idx3 - width_;
                //     const float32x2_t v1 = vld1_f32(bilinear_values_ + i + offset);
                //     const float32x2_t v2 = vld1_f32(bilinear_values_ + i + offset + width_);

                //     // const float32x2_t im1 = vld1_f32(image_target + idx1);
                //     // const float32x2_t im2 = vld1_f32(image_target + idx3);
                //     const float32x2_t im1 = vld1_f32(image_target);
                //     const float32x2_t im2 = vld1_f32(image_target);
                //     // float32x2_t sum = vadd_f32(v1, im1);
                //     // vst1_f32(image_target + idx1, sum);
                //     // sum = vadd_f32(v2, im2);
                //     // vst1_f32(image_target + idx3, sum);
                //     float32x2_t sum = vadd_f32(v1, im1);
                //     vst1_f32(image_target , sum);
                //     sum = vadd_f32(v2, im2);
                //     vst1_f32(image_target , sum);
                }
            }
        }
        nvtxRangePop();

        // Calculate contrast and if needed jacobian

        // nvtxRangePushA("contrast"); // Begins NVTX range

        getContrastDelBatchReduce(image_, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_, residuals, gradient, height_, width_,
                                  contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, means_, contrast_block_sum_cpu_);
        cudaMemsetAsync(image_, 0, (height_) * (width_) * sizeof(float));
        cudaMemsetAsync(image_del_theta_x_, 0, (height_) * (width_) * sizeof(float));
        cudaMemsetAsync(image_del_theta_y_, 0, (height_) * (width_) * sizeof(float));
        cudaMemsetAsync(image_del_theta_z_, 0, (height_) * (width_) * sizeof(float));
        // nvtxRangePop();
        return true;
    }
    int NumParameters() const override { return 3; }
    void GenerateImage(const double *const rotations, uint8_t *output_image)
    {
        float *image;
        cudaAllocMapped(&image, sizeof(float) * height_ * width_);
        std::fill_n(image, (height_) * (width_), 0);
        fillImageKronecker(height_, width_, num_events_, x_prime_, y_prime_, image);
        cudaDeviceSynchronize();
        float maximum = getMax(image, height_, width_);
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
            }
        }
        cudaFree(image);
    };

    int f_count = 0;
    int g_count = 0;

// private:
    float *x_unprojected_ = NULL;
    float *y_unprojected_ = NULL;
    float *x_prime_ = NULL;
    float *y_prime_ = NULL;
    float *bilinear_values_ = NULL;
    float *t_ = NULL;
    int height_;
    int width_;
    int num_events_;
    float *image_ = NULL;
    float *image_del_theta_x_ = NULL;
    float *image_del_theta_y_ = NULL;
    float *image_del_theta_z_ = NULL;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    int cub_temp_size_;
    float *temp_storage_;

    float *contrast_block_sum_;
    float *contrast_del_x_block_sum_;
    float *contrast_del_y_block_sum_;
    float *contrast_del_z_block_sum_;
    float *means_;
    float *contrast_block_sum_cpu_;
};

#endif // MC_GRADIENT_BILINEAR_H