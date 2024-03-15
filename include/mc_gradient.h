#ifndef MC_GRADIENT_H
#define MC_GRADIENT_H
#include "ceres/ceres.h"
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

#include <fstream>
#include <iostream>
#include <vector>
// A CostFunction implementing motion compensation then calculating contrast, as well as the jacobian.
class McGradient final : public ceres::FirstOrderFunction
{

public:
    McGradient(const double fx, const double fy, const double cx, const double cy,
               std::vector<double> &x, std::vector<double> &y, std::vector<double> &t, const int height, const int width, const int num_events) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width), num_events_(num_events)
    {
        // create pinned memory for x,y,t,image,image dels
        cudaMalloc(&x_unprojected_, num_events_ * sizeof(double));
        cudaMalloc(&y_unprojected_, num_events_ * sizeof(double));
        cudaMalloc(&x_prime_, num_events_ * sizeof(double));
        cudaMalloc(&y_prime_, num_events_ * sizeof(double));
        cudaMalloc(&t_, num_events_ * sizeof(double));
        cudaMalloc(&image_, (height_ + 6) * (width_ + 6) * sizeof(double));
        cudaMalloc(&image_del_theta_x_, (height_ + 6) * (width_ + 6) * sizeof(double));
        cudaMalloc(&image_del_theta_y_, (height_ + 6) * (width_ + 6) * sizeof(double));
        cudaMalloc(&image_del_theta_z_, (height_ + 6) * (width_ + 6) * sizeof(double));
        image_empty_=new double[(height_ + 6) * (width_ + 6)];
        std::fill_n(image_empty_,(height_ + 6) * (width_ + 6),0);
        
        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // double scale=t[num_events-1]-t[0];
        double scale = 1e6;
        double t_cpu[num_events_];
        for (int i = 1; i < num_events_; i++)
        {
            t_cpu[i] = (t[i] - t[0]) / scale;
        }
        cudaMemcpy(t_,t_cpu,num_events_*sizeof(double),cudaMemcpyHostToDevice);

        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        double x_unprojected_cpu[num_events_];
        double y_unprojected_cpu[num_events_];
        for (int i = 0; i < num_events_; i++)
        {
            x_unprojected_cpu[i] = (x[i] - cx) / fx;
            y_unprojected_cpu[i] = (y[i] - cy) / fy;
        }
        cudaMemcpy(x_unprojected_,x_unprojected_cpu,num_events_*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(y_unprojected_,y_unprojected_cpu,num_events_*sizeof(double),cudaMemcpyHostToDevice);
        cub_temp_size_=getCubSize(image_,height_,width_);
    }
    void tryCudaAllocMapped(double **ptr, size_t size, std::string ptr_name)
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
        cudaMemset(image_,0,(height_ + 6) * (width_ + 6)*sizeof(double));
        cudaMemset(image_del_theta_x_,0,(height_ + 6) * (width_ + 6)*sizeof(double));
        cudaMemset(image_del_theta_y_,0,(height_ + 6) * (width_ + 6)*sizeof(double));
        cudaMemset(image_del_theta_z_,0,(height_ + 6) * (width_ + 6)*sizeof(double));
        bool do_jacobian = gradient != nullptr;
        // Populate image
        // std::cout << "filling" << std::endl;
        // std::cout << "rotations " << parameters[0] << " " << parameters[1] << " " << parameters[2] << std::endl;
        
        cudaEventRecord(start);
        fillImage(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], do_jacobian, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time_ms;
        cudaEventElapsedTime(&time_ms,start,stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // std::cout<<"fillImage time: "<<time_ms<<std::endl;

        
        // remove mean from image
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        subtractMean(image_, height_, width_,cub_temp_size_);
        
        cudaDeviceSynchronize();
        if (do_jacobian)
        {
            subtractMean(image_del_theta_x_, height_, width_,cub_temp_size_);
            subtractMean(image_del_theta_y_, height_, width_,cub_temp_size_);
            subtractMean(image_del_theta_z_, height_, width_,cub_temp_size_);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms,start,stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // std::cout<<"subtractmean time: "<<time_ms<<std::endl;

        // Calculate contrast and if needed jacobian
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        residuals[0] = -getContrast(image_, height_, width_,cub_temp_size_);
        if (do_jacobian)
        {
            gradient[0] = -getContrastDel(image_, image_del_theta_x_, height_, width_,cub_temp_size_);
            gradient[1] = -getContrastDel(image_, image_del_theta_y_, height_, width_,cub_temp_size_);
            gradient[2] = -getContrastDel(image_, image_del_theta_z_, height_, width_,cub_temp_size_);
            cudaDeviceSynchronize();

            // std::cout << "gradient " << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms,start,stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // std::cout<<"getcontrast time: "<<time_ms<<std::endl;
        // std::cout << "residual " << residuals[0] << std::endl;
        return true;
    }
    int NumParameters() const override { return 3; }
    void GenerateImage(const double *const rotations, uint8_t *output_image,double contrast)
    {
        double image_cpu[(height_ + 6) * (width_ + 6)];
        std::fill_n(image_cpu,(height_ + 6) * (width_ + 6),0);
        cudaMemset(image_,0,(height_ + 6) * (width_ + 6)*sizeof(double));
        fillImageKronecker(height_,  width_, num_events_,x_prime_, y_prime_, image_);
        cudaMemcpy(image_cpu,image_,(height_ + 6) * (width_ + 6)*sizeof(double),cudaMemcpyDeviceToHost);
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i*width_+j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image_cpu[(i+3) * (width_ + 6) + j+3] / sqrt(contrast))));  
            }
        }
    };

private:
    double *x_unprojected_ = NULL;
    double *y_unprojected_ = NULL;
    double *x_prime_ = NULL;
    double *y_prime_ = NULL;
    double *t_ = NULL;
    int height_;
    int width_;
    int num_events_;
    double *image_empty_ = NULL;
    double *image_ = NULL;
    double *image_del_theta_x_ = NULL;
    double *image_del_theta_y_ = NULL;
    double *image_del_theta_z_ = NULL;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    int cub_temp_size_;
};

#endif // MC_GRADIENT_H