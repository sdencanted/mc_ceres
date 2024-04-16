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
#include <thread>
#include <condition_variable>
#include <iostream>
#include <mutex>
void notify(std::shared_ptr<std::condition_variable> cv)
{
    cv->notify_one();
}
// A CostFunction implementing motion compensation then calculating contrast, as well as the jacobian.
class McGradientBilinear final : public ceres::FirstOrderFunction
{

public:
    ~McGradientBilinear()
    {
        cudaFree(x_unprojected_);
        cudaFree(y_unprojected_);
        cudaFree(x_prime_);
        cudaFree(y_prime_);
        cudaFree(t_);
        cudaFree(image_);
        
        cudaStreamDestroy(stream_[0]);
        cudaStreamDestroy(stream_[1]);
        // cudaFree(image_del_theta_x_);
        // cudaFree(image_del_theta_y_);
        // cudaFree(image_del_theta_z_);

        checkCudaErrors(cudaFreeHost(contrast_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_x_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_y_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_z_block_sum_));
        checkCudaErrors(cudaFreeHost(contrast_block_sum_cpu_));
        checkCudaErrors(cudaFreeHost(means_));

        // running = false;
        // cv_->notify_one();
        // if (memset_thread_->joinable())
        //     memset_thread_->join();
    }
    McGradientBilinear(const float fx, const float fy, const float cx, const float cy,
                       std::vector<float> &x, std::vector<float> &y, std::vector<float> &t, const int height, const int width, const int num_events) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width), num_events_(num_events)
    {
        cv_ = std::make_shared<std::condition_variable>();
        // create pinned memory for x,y,t,image,image dels
        cudaMallocHost(&x_unprojected_, num_events_ * sizeof(float));
        cudaMallocHost(&y_unprojected_, num_events_ * sizeof(float));
        cudaMalloc(&x_prime_, num_events_ * sizeof(float));
        cudaMalloc(&y_prime_, num_events_ * sizeof(float));
        cudaMallocHost(&t_, num_events_ * sizeof(float));
        cudaMalloc(&image_, (height_) * (width_) * sizeof(float) * 4);
        // cudaMalloc(&image_del_theta_x_, (height_) * (width_) * sizeof(float));
        // cudaMalloc(&image_del_theta_y_, (height_) * (width_) * sizeof(float));
        // cudaMalloc(&image_del_theta_z_, (height_) * (width_) * sizeof(float));

        cudaMemsetAsync(image_, 0, (height_) * (width_) * sizeof(float) * 4);
        // cudaMemsetAsync(image_del_theta_x_, 0, (height_) * (width_) * sizeof(float));
        // cudaMemsetAsync(image_del_theta_y_, 0, (height_) * (width_) * sizeof(float));
        // cudaMemsetAsync(image_del_theta_z_, 0, (height_) * (width_) * sizeof(float));
        checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, 128 * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, 128 * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, 128 * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, 128 * sizeof(float)));
        cudaMemsetAsync(contrast_block_sum_, 0, 128 * sizeof(float));
        cudaMemsetAsync(contrast_del_x_block_sum_, 0, 128 * sizeof(float));
        cudaMemsetAsync(contrast_del_y_block_sum_, 0, 128 * sizeof(float));
        cudaMemsetAsync(contrast_del_z_block_sum_, 0, 128 * sizeof(float));
        checkCudaErrors(cudaMallocHost(&means_, 4 * sizeof(float)));
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

        cudaStreamCreate(&stream_[0]);
        cudaStreamCreate(&stream_[1]);
        // cudaDeviceSynchronize();
        // memset_thread_ = std::make_shared<std::thread>(&McGradientBilinear::memsetFunc, this);
        // memset_thread_->detach();
    }
    // void memsetFunc()
    // {
    //     // std::mutex m;
    //     while (running)
    //     {
    //         std::unique_lock lk(m_);
    //         cv_->wait(lk);
    //         if (!running)
    //             break;
    //         cudaMemsetAsync(this->image_, 0, (this->height_) * (this->width_) * sizeof(float) * 4);
    //         lk.unlock();
    //     }
    // }
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
        cudaFree(x_prime_);
        cudaFree(y_prime_);
        cudaFree(t_);

        cudaMalloc(&x_unprojected_, num_events_ * sizeof(float));
        cudaMalloc(&y_unprojected_, num_events_ * sizeof(float));
        cudaMalloc(&x_prime_, num_events_ * sizeof(float));
        cudaMalloc(&y_prime_, num_events_ * sizeof(float));
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
        // Populate image
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);

        getContrastDelBatchReduce(image_, residuals, gradient, height_, width_,
                                  contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, means_, contrast_block_sum_cpu_, num_events_,stream_);

        // cudaMemsetAsync(this->image_, 0, (this->height_) * (this->width_) * sizeof(float)*4);

        // cv_->notify_one();
        // cudaMemsetAsync(image_, 0, (height_) * (width_) * sizeof(float)*4);
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

private:
    float *x_unprojected_ = NULL;
    float *y_unprojected_ = NULL;
    float *x_prime_ = NULL;
    float *y_prime_ = NULL;
    float *t_ = NULL;
    int height_;
    int width_;
    int num_events_;
    float *image_ = NULL;
    // float *image_del_theta_x_ = NULL;
    // float *image_del_theta_y_ = NULL;
    // float *image_del_theta_z_ = NULL;
    float fx_;
    float fy_;
    float cx_;
    float cy_;

    float *contrast_block_sum_;
    float *contrast_del_x_block_sum_;
    float *contrast_del_y_block_sum_;
    float *contrast_del_z_block_sum_;
    float *means_;
    float *contrast_block_sum_cpu_;
    std::shared_ptr<std::thread> memset_thread_;
    std::mutex m_;
    std::shared_ptr<std::condition_variable> cv_;
    bool running = true;
    cudaStream_t stream_[2];
};

#endif // MC_GRADIENT_BILINEAR_H