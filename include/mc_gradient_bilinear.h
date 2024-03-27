#ifndef MC_GRADIENT_BILINEAR_H
#define MC_GRADIENT_BILINEAR_H
#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
#include "glog/logging.h"

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
        cudaFree(image_del_theta_x_);
        cudaFree(image_del_theta_y_);
        cudaFree(image_del_theta_z_);
    }
    McGradientBilinear(const float fx, const float fy, const float cx, const float cy,
                       std::vector<float> &x, std::vector<float> &y, std::vector<float> &t, const int height, const int width, const int num_events, bool middle_timestamp, bool split_func = false) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width), num_events_(num_events), split_func_(split_func)
    {
        // create pinned memory for x,y,t,image,image dels
        cudaMalloc(&x_unprojected_, num_events_ * sizeof(float));
        cudaMalloc(&y_unprojected_, num_events_ * sizeof(float));
        cudaMalloc(&x_prime_, num_events_ * sizeof(float));
        cudaMalloc(&y_prime_, num_events_ * sizeof(float));
        cudaMalloc(&t_, num_events_ * sizeof(float));
        // cudaMalloc(&image_, (height_) * (width_) * sizeof(float));
        // cudaMalloc(&image_del_theta_x_, (height_) * (width_) * sizeof(float));
        // cudaMalloc(&image_del_theta_y_, (height_) * (width_) * sizeof(float));
        // cudaMalloc(&image_del_theta_z_, (height_) * (width_) * sizeof(float));
        cudaMalloc(&image_,57*768 * sizeof(float));
        cudaMalloc(&image_del_theta_x_, 57*768 * sizeof(float));
        cudaMalloc(&image_del_theta_y_, 57*768 * sizeof(float));
        cudaMalloc(&image_del_theta_z_, 57*768 * sizeof(float));

        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // float scale=t[num_events-1]-t[0];
        float scale = 1e6;
        float t_cpu[num_events_];
        // find the middle t
        float middle_t = (t[num_events_ - 1] + t[0]) / 2;

        for (int i = 1; i < num_events_; i++)
        {
            if (middle_timestamp)
            {
                t_cpu[i] = (t[i] - middle_t) / scale;
            }
            else
            {
                t_cpu[i] = (t[i] - t[0]) / scale;
            }
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
    }
    void tryCudaAllocMapped(float **ptr, size_t size, std::string ptr_name)
    {
        std::cout << "allocating cuda mem for " << ptr_name << std::endl;
        if (!cudaAllocMapped(ptr, size))
        {
            std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
        }
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const override
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaMemset(image_, 0, (height_) * (width_) * sizeof(float));
        cudaMemset(image_del_theta_x_, 0, (height_) * (width_) * sizeof(float));
        cudaMemset(image_del_theta_y_, 0, (height_) * (width_) * sizeof(float));
        cudaMemset(image_del_theta_z_, 0, (height_) * (width_) * sizeof(float));
        bool do_jacobian = gradient != nullptr;
        // Populate image
        // std::cout << "filling" << std::endl;
        // std::cout << "rotations " << parameters[0] << " " << parameters[1] << " " << parameters[2] << std::endl;

        cudaEventRecord(start);
        if (split_func_)
        {
            fillImageBilinearSeparate(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], do_jacobian, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_);
        }
        else
        {
            fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], do_jacobian, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_);
        }
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // std::cout<<"fillImage time: "<<time_ms<<std::endl;

        // remove mean from image
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);
        // subtractMean(image_, height_, width_ , cub_temp_size_);
        // if (do_jacobian)
        // {
        //     subtractMean(image_del_theta_x_, height_ , width_ , cub_temp_size_);
        //     subtractMean(image_del_theta_y_, height_ , width_ , cub_temp_size_);
        //     subtractMean(image_del_theta_z_, height_, width_ , cub_temp_size_);
        // }
        // cudaDeviceSynchronize();
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&time_ms, start, stop);
        // cudaEventDestroy(start);
        // cudaEventDestroy(stop);
        // std::cout<<"subtractmean time: "<<time_ms<<std::endl;

        // Calculate contrast and if needed jacobian
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        if (do_jacobian){
            getContrastDelBatch(image_,image_del_theta_x_,image_del_theta_y_,image_del_theta_z_,residuals,gradient,height_,width_,cub_temp_size_);
        }
        else{
            residuals[0] = -getContrast(image_, height_, width_, cub_temp_size_);
        }
        // if (do_jacobian)
        // {
        //     gradient[0] = -getContrastDel(image_, image_del_theta_x_, height_, width_, cub_temp_size_);
        //     gradient[1] = -getContrastDel(image_, image_del_theta_y_, height_, width_, cub_temp_size_);
        //     gradient[2] = -getContrastDel(image_, image_del_theta_z_, height_, width_, cub_temp_size_);

        //     // std::cout << "gradient " << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
        // }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // std::cout<<"getcontrast time: "<<time_ms<<std::endl;
        // std::cout << "residual " << residuals[0] << std::endl;
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
    bool split_func_ = false;
    float *x_unprojected_ = NULL;
    float *y_unprojected_ = NULL;
    float *x_prime_ = NULL;
    float *y_prime_ = NULL;
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
};

#endif // MC_GRADIENT_BILINEAR_H