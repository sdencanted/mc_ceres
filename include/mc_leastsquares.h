#ifndef MC_JACOBIAN_H
#define MC_JACOBIAN_H
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


template <typename SCALAR>
void tryCudaAllocMapped(SCALAR **ptr, size_t size, std::string ptr_name)
{
    std::cout << "allocating cuda mem for " << ptr_name << std::endl;
    if (!cudaAllocMapped((void**)ptr, size))
    {
        std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
    }
}
// A CostFunction implementing motion compensation then calculating contrast, as well as the jacobian.
class McCostFunction
    : public ceres::SizedCostFunction<1 /* negative of contrast */,
                                      3 /* 3 axis rotation */>
{
public:
    McCostFunction(const double fx, const double fy, const double cx, const double cy,
                   std::vector<double> &x, std::vector<double> &y, std::vector<double> &t, const int height, const int width, const int num_events) : SizedCostFunction(), fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width), num_events_(num_events)
    {
        // create pinned memory for x,y,t,image,image dels
        tryCudaAllocMapped(&x_unprojected_, num_events_ * sizeof(double), "x_unprojected_");
        tryCudaAllocMapped(&y_unprojected_, num_events_ * sizeof(double), "y_unprojected_");
        tryCudaAllocMapped(&x_prime_, num_events_ * sizeof(double), "x_prime_");
        tryCudaAllocMapped(&y_prime_, num_events_ * sizeof(double), "y_prime_");
        tryCudaAllocMapped(&t_, num_events_ * sizeof(double), "t");
        tryCudaAllocMapped(&image_, (height_ + 6) * (width_ + 6) * sizeof(double), "image");
        tryCudaAllocMapped(&image_del_theta_x_, (height_ + 6) * (width_ + 6) * sizeof(double), "image_del_x");
        tryCudaAllocMapped(&image_del_theta_y_, (height_ + 6) * (width_ + 6) * sizeof(double), "image_del_y");
        tryCudaAllocMapped(&image_del_theta_z_, (height_ + 6) * (width_ + 6) * sizeof(double), "image_del_z");

        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // double scale=t[num_events-1]-t[0];
        double scale = 1e6;
        std::cout << scale << std::endl;
        for (int i = 1; i < num_events_; i++)
        {
            t_[i] = (t[i] - t[0]) / scale;
        }

        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        for (int i = 0; i < num_events_; i++)
        {
            x_unprojected_[i] = (x[i] - cx) / fx;
            y_unprojected_[i] = (y[i] - cy) / fy;
            // x_unprojected_[i] = x[i];
            // y_unprojected_[i] = y[i];
        }
    }
    bool Evaluate(double const *const *parameters,
                  double *residuals,
                  double **jacobians) const override
    {

        std::fill_n(image_, (height_ + 6) * (width_ + 6), 0);
        std::fill_n(image_del_theta_x_, (height_ + 6) * (width_ + 6), 0);
        std::fill_n(image_del_theta_y_, (height_ + 6) * (width_ + 6), 0);
        std::fill_n(image_del_theta_z_, (height_ + 6) * (width_ + 6), 0);
        cudaDeviceSynchronize();
        bool do_jacobian = jacobians != nullptr;
        // Populate image
        // std::cout << "filling" << std::endl;
        std::cout << "rotations " << parameters[0][0] << " " << parameters[0][1] << " " << parameters[0][2] << std::endl;
        cudaDeviceSynchronize();
        fillImage(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0][0], parameters[0][1], parameters[0][2], do_jacobian, image_del_theta_x_, image_del_theta_y_, image_del_theta_z_);
        cudaDeviceSynchronize();
        if (abs(parameters[0][0] - 1e-7) < 1e-8)
        {
            std::ofstream outfile("out.csv", std::ios::out);
            for (int i = 0; i < height_; i++)
            {

                for (int j = 0; j < width_; j++)
                {
                    outfile << image_[(i + 3) * (width_ + 6) + (j + 3)];
                    if (j < width_ - 1)
                    {
                        outfile << ",";
                    }
                }
                outfile << std::endl;
            }
            outfile.close();

            // std::ofstream outfile2("xy.csv",std::ios::out);
            // for(int j=0;j<num_events_;j++){
            //     outfile2<<x_prime_[j]<<","<<y_prime_[j]<<std::endl;
            // }
            // outfile2.close();
        }
        // remove mean from image
        // std::cout << "subtracting mean" << std::endl;
        subtractMean(image_, height_, width_);
        cudaDeviceSynchronize();
        std::cout << getMean(image_, height_, width_) << std::endl;
        if (do_jacobian)
        {

            if (abs(parameters[0][0] - 1e-7) < 1e-8)
            {
                std::ofstream outfile2("out2.csv", std::ios::out);
                for (int i = 0; i < height_; i++)
                {

                    for (int j = 0; j < width_; j++)
                    {
                        outfile2 << image_del_theta_x_[(i + 3) * (width_ + 6) + (j + 3)];
                        if (j < width_ - 1)
                        {
                            outfile2 << ",";
                        }
                    }
                    outfile2 << std::endl;
                }
                outfile2.close();
            }
            subtractMean(image_del_theta_x_, height_, width_);
            subtractMean(image_del_theta_y_, height_, width_);
            subtractMean(image_del_theta_z_, height_, width_);
            cudaDeviceSynchronize();
        }

        // Calculate contrast and if needed jacobian
        residuals[0] = -getContrast(image_, height_, width_);
        if (do_jacobian)
        {
            jacobians[0][0] = -getContrastDel(image_, image_del_theta_x_, height_, width_);
            jacobians[0][1] = -getContrastDel(image_, image_del_theta_y_, height_, width_);
            jacobians[0][2] = -getContrastDel(image_, image_del_theta_z_, height_, width_);
            cudaDeviceSynchronize();

            std::cout << "jacobians " << jacobians[0][0] << " " << jacobians[0][1] << " " << jacobians[0][2] << std::endl;
        }
        cudaDeviceSynchronize();
        std::cout << "residual " << residuals[0] << std::endl;
        return true;
    }

private:
    double *x_unprojected_ = NULL;
    double *y_unprojected_ = NULL;
    double *x_prime_ = NULL;
    double *y_prime_ = NULL;
    double *t_ = NULL;
    int height_;
    int width_;
    int num_events_;
    double *image_ = NULL;
    double *image_del_theta_x_ = NULL;
    double *image_del_theta_y_ = NULL;
    double *image_del_theta_z_ = NULL;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
};
#endif // MC_GRADIENT_H