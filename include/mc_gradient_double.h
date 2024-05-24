#ifndef MC_GRADIENT_DOUBLE_H
#define MC_GRADIENT_DOUBLE_H
#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/core/multi_stream_slicer.hpp>
// #include "glog/logging.h"

// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_double.h"

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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/async/copy.h>
void notify(std::shared_ptr<std::condition_variable> cv)
{
    cv->notify_one();
}
// A CostFunction implementing motion compensation then calculating contrast, as well as the jacobian.
class McGradient final : public ceres::FirstOrderFunction
{

public:
    ~McGradient()
    {
        // checkCudaErrors(cudaFreeHost(x_unprojected_));
        // checkCudaErrors(cudaFreeHost(y_unprojected_));
        // checkCudaErrors(cudaFree(x_prime_));
        // checkCudaErrors(cudaFree(y_prime_));
        // checkCudaErrors(cudaFreeHost(t_));
        // checkCudaErrors(cudaFreeHost(x_));
        // checkCudaErrors(cudaFreeHost(y_));
        // cudaStreamDestroy(stream_[0]);
        // cudaStreamDestroy(stream_[1]);

        // checkCudaErrors(cudaFreeHost(contrast_block_sum_));
        // checkCudaErrors(cudaFree(contrast_del_x_block_sum_));
        // checkCudaErrors(cudaFree(contrast_del_y_block_sum_));
        // checkCudaErrors(cudaFree(contrast_del_z_block_sum_));
        // checkCudaErrors(cudaFreeHost(means_));

        // // running = false;
        // // cv_->notify_one();
        // // if (memset_thread_->joinable())
        // //     memset_thread_->join();
    }
    McGradient(const double fx, const double fy, const double cx, const double cy, const int height, const int width) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width)
    {
        // cv_ = std::make_shared<std::condition_variable>();

        cudaStreamCreate(&stream_[0]);
        cudaStreamCreate(&stream_[1]);
        // create pinned memory for x,y,t,image,image dels

        checkCudaErrors(cudaMalloc(&image_, (height_) * (width_) * sizeof(double) * 4));
        int gridSize = std::min(512, (height * width + 512 - 1) / 512);
        // checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, 128 * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, 128 * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, 128 * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, 128 * sizeof(double)));
        checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, gridSize * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, gridSize * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, gridSize * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, gridSize * sizeof(double)));
        checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(double)));

        cudaMemsetAsync(image_, 0, (height_) * (width_) * sizeof(double) * 4);
        cudaMemsetAsync(contrast_block_sum_, 0, gridSize * sizeof(double));
        cudaMemsetAsync(contrast_del_x_block_sum_, 0, gridSize * sizeof(double));
        cudaMemsetAsync(contrast_del_y_block_sum_, 0, gridSize * sizeof(double));
        cudaMemsetAsync(contrast_del_z_block_sum_, 0, gridSize * sizeof(double));

        // ReplaceData(x, y, t, num_events_);

        // for uncompensated image
    }
    void tryCudaAllocMapped(double **ptr, size_t size, std::string ptr_name)
    {
        std::cout << "allocating cuda mem for " << ptr_name << std::endl;
        if (!cudaAllocMapped(ptr, size))
        {
            std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
        }
    }
    void ReplaceData(std::vector<double> &x, std::vector<double> &y, std::vector<double> &t, const int num_events)
    {

        nvtx3::scoped_range r{"replace data"};
        num_events_ = num_events;
        if (!allocated_ || max_num_events_ < num_events_)
        {
            max_num_events_ = std::max(num_events_, 30000);
            if (allocated_)
            {
                checkCudaErrors(cudaFreeHost(x_unprojected_));
                checkCudaErrors(cudaFreeHost(y_unprojected_));
                checkCudaErrors(cudaFree(x_prime_));
                checkCudaErrors(cudaFree(y_prime_));
                checkCudaErrors(cudaFreeHost(t_));
                checkCudaErrors(cudaFreeHost(x_));
                checkCudaErrors(cudaFreeHost(y_));
            }

            checkCudaErrors(cudaMallocHost(&x_unprojected_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMallocHost(&y_unprojected_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMalloc(&x_prime_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMalloc(&y_prime_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMallocHost(&x_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMallocHost(&y_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMallocHost(&t_, max_num_events_ * sizeof(double)));

            // cudaMalloc(&x_, num_events_ * sizeof(double));
            // cudaMalloc(&y_, num_events_ * sizeof(double));
            cudaMemcpyAsync(x_, x.data(), num_events_ * sizeof(double), cudaMemcpyDefault);
            cudaMemcpyAsync(y_, y.data(), num_events_ * sizeof(double), cudaMemcpyDefault);
            allocated_ = true;
        }
        // // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // // double scale=t[num_events-1]-t[0];
        // double scale = 1e6;
        // // find the middle t
        // double middle_t = (t[num_events_ - 1] + t[0]) / 2;
        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        cudaMemcpyAsync(t_, t.data(), num_events_ * sizeof(double), cudaMemcpyDefault);
        for (int i = 0; i < num_events_; i++)
        {
            // t_[i] = (t[i] - middle_t) / scale;
            x_unprojected_[i] = (x[i] - cx_) / fx_;
            y_unprojected_[i] = (y[i] - cy_) / fy_;
        }
    }

    // void ReplaceData(std::vector<double> &x, std::vector<double> &y, std::vector<double> &t, const int num_events)
    // {
    //     nvtx3::scoped_range r{"replace data"};
    //     num_events_ = num_events;
    //     if (!allocated_ || max_num_events_ < num_events_)
    //     {
    //         max_num_events_ = std::max(num_events_, 30000);
    //         if (allocated_)
    //         {
    //             checkCudaErrors(cudaFreeHost(x_unprojected_));
    //             checkCudaErrors(cudaFreeHost(y_unprojected_));
    //             checkCudaErrors(cudaFree(x_prime_));
    //             checkCudaErrors(cudaFree(y_prime_));
    //             checkCudaErrors(cudaFreeHost(t_));
    //             checkCudaErrors(cudaFreeHost(x_));
    //             checkCudaErrors(cudaFreeHost(y_));
    //         }

    //         checkCudaErrors(cudaMallocHost(&x_unprojected_, max_num_events_ * sizeof(double)));
    //         checkCudaErrors(cudaMallocHost(&y_unprojected_, max_num_events_ * sizeof(double)));
    //         checkCudaErrors(cudaMalloc(&x_prime_, max_num_events_ * sizeof(double)));
    //         checkCudaErrors(cudaMalloc(&y_prime_, max_num_events_ * sizeof(double)));
    //         checkCudaErrors(cudaMallocHost(&x_, max_num_events_ * sizeof(double)));
    //         checkCudaErrors(cudaMallocHost(&y_, max_num_events_ * sizeof(double)));
    //         checkCudaErrors(cudaMallocHost(&t_, max_num_events_ * sizeof(double)));

    //         // cudaMalloc(&x_, num_events_ * sizeof(double));
    //         // cudaMalloc(&y_, num_events_ * sizeof(double));
    //         allocated_ = true;
    //     }
    //     cudaMemcpyAsync(x_, x.data(), num_events_ * sizeof(double), cudaMemcpyDefault);
    //     cudaMemcpyAsync(y_, y.data(), num_events_ * sizeof(double), cudaMemcpyDefault);
    //     // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
    //     // double scale=t[num_events-1]-t[0];
    //     // find the middle t
    //     double middle_t = (t.back() + t.front()) / 2;
    //     // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
    //     for (int i = 0; i < num_events_; i++)
    //     {
    //         t_[i] = (t[i] - middle_t);
    //         x_unprojected_[i] = (x[i] - cx_) / fx_;
    //         y_unprojected_[i] = (y[i] - cy_) / fy_;
    //     }
    //     // std::cout<<"time "<<t_[0]<<" "<<t_[num_events_-1]<<std::endl;
    // }
    void reset()
    {
        // cudaStreamDestroy(stream_[0]);
        // cudaStreamDestroy(stream_[1]);
        // cudaStreamCreate(&stream_[0]);
        // cudaStreamCreate(&stream_[1]);
        // create pinned memory for x,y,t,image,image dels
        // cudaFree(image_);
        // checkCudaErrors(cudaMalloc(&image_, (height_) * (width_) * sizeof(double) * 4));
        int gridSize = std::min(512, (height_ * width_ + 512 - 1) / 512);

        // cudaFreeHost(contrast_block_sum_);
        // cudaFree(contrast_del_x_block_sum_);
        // cudaFree(contrast_del_y_block_sum_);
        // cudaFree(contrast_del_z_block_sum_);
        // cudaFree(means_);
        // checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, gridSize * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, gridSize * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, gridSize * sizeof(double)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, gridSize * sizeof(double)));
        // checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(double)));

        cudaMemset(image_, 0, (height_) * (width_) * sizeof(double) * 4);
        // cudaMemset(contrast_block_sum_, 0, gridSize * sizeof(double));
        // cudaMemset(contrast_del_x_block_sum_, 0, gridSize * sizeof(double));
        // cudaMemset(contrast_del_y_block_sum_, 0, gridSize * sizeof(double));
        // cudaMemset(contrast_del_z_block_sum_, 0, gridSize * sizeof(double));

        // ReplaceData(x, y, t, num_events_);

        // for uncompensated image
    }
    void allocate()
    {
        cudaMallocHost(&x_unprojected_, max_num_events_ * sizeof(double));
        cudaMallocHost(&y_unprojected_, max_num_events_ * sizeof(double));
        cudaMalloc(&x_prime_, max_num_events_ * sizeof(double));
        cudaMalloc(&y_prime_, max_num_events_ * sizeof(double));
        checkCudaErrors(cudaMallocHost(&x_, max_num_events_ * sizeof(double)));
        checkCudaErrors(cudaMallocHost(&y_, max_num_events_ * sizeof(double)));
        cudaMallocHost(&t_, max_num_events_ * sizeof(double));
        allocated_ = true;
    }
    void ReplaceData(const dv::AddressableEventStorage<dv::Event, dv::EventPacket> &data)
    {
        nvtx3::scoped_range r{"replace data"};
        num_events_ = data.size();
        if (!allocated_ || max_num_events_ < num_events_)
        {
            max_num_events_ = std::max(num_events_, 30000);
            if (allocated_)
            {
                cudaFreeHost(x_unprojected_);
                cudaFreeHost(y_unprojected_);
                cudaFree(x_prime_);
                cudaFree(y_prime_);
                checkCudaErrors(cudaFreeHost(x_));
                checkCudaErrors(cudaFreeHost(y_));
                cudaFreeHost(t_);
            }

            cudaMallocHost(&x_unprojected_, max_num_events_ * sizeof(double));
            cudaMallocHost(&y_unprojected_, max_num_events_ * sizeof(double));
            cudaMalloc(&x_prime_, max_num_events_ * sizeof(double));
            cudaMalloc(&y_prime_, max_num_events_ * sizeof(double));
            checkCudaErrors(cudaMallocHost(&x_, max_num_events_ * sizeof(double)));
            checkCudaErrors(cudaMallocHost(&y_, max_num_events_ * sizeof(double)));
            cudaMallocHost(&t_, max_num_events_ * sizeof(double));
            allocated_ = true;
        }
        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // double scale=t[num_events-1]-t[0];
        double scale = 1e6;
        // find the middle t

        int64_t middle_t = (data.back().timestamp() + data.front().timestamp()) / 2;
        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        int i = 0;

        // cudaMemcpyAsync(x_,coords.data(),num_events_*sizeof(double),cudaMemcpyDefault);
        // cudaMemcpyAsync(y_,coords.data()+num_events_,num_events_*sizeof(double),cudaMemcpyDefault);
        for (auto event : data)
        {
            t_[i] = (event.timestamp() - middle_t) / scale;
            // std::cout<<t_[i]<<" "<<event.timestamp()<<" "<<middle_t<<" "<<(event.timestamp() - middle_t)<<" "<<scale<<std::endl;
            x_unprojected_[i] = (event.x() - cx_) / fx_;
            y_unprojected_[i] = (event.y() - cy_) / fy_;
            x_[i] = event.x();
            y_[i] = event.y();
            i++;
        }
        // std::cout << i << " events loaded" << t_[0] << " " << t_[num_events_ - 1] << " " << data.back().timestamp() << " " << data.front().timestamp() << std::endl;
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const override
    {
        
        // return EvaluateCpu(parameters,residuals,gradient);
        nvtx3::scoped_range r{"Evaluate"};
        fillImage(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);

        // SEMI CPU
        // double image_contrast = 0;
        // double image_contrast_del_theta_x = 0;
        // double image_contrast_del_theta_y = 0;
        // double image_contrast_del_theta_z = 0;
        // double means[4]={0};
        // cudaDeviceSynchronize();
        // for (int i=0;i<4;i++){
        //     means[i]=thrustMean(image_+height_*width_*i,height_,width_);
        //     std::cout<<means[i]<<std::endl;
        // }
        // double image[height_ * width_ * 4];
        // cudaMemcpy(image, image_, sizeof(double) * height_ * width_ * 4, cudaMemcpyDefault);
        // for (int row = 0; row < height_; row++)
        // {
        //     for (int col = 0; col < width_; col++)
        //     {
        //         int idx = row * width_ + col;
        //         double image_norm = image[idx] - means[0];
        //         double image_norm_x = image[idx + height_ * width_ * 1] - means[1];
        //         double image_norm_y = image[idx + height_ * width_ * 2] - means[2];
        //         double image_norm_z = image[idx + height_ * width_ * 3] - means[3];
        //         image_contrast += image_norm * image_norm;
        //         image_contrast_del_theta_x += image_norm_x * image_norm;
        //         image_contrast_del_theta_y += image_norm_y * image_norm;
        //         image_contrast_del_theta_z += image_norm_z * image_norm;
        //     }
        // }

        // int num_el = height_ * width_;
        // residuals[0] = -image_contrast / num_el;
        // gradient[0] = -2 * image_contrast_del_theta_x / num_el;
        // gradient[1] = -2 * image_contrast_del_theta_y / num_el;
        // gradient[2] = -2 * image_contrast_del_theta_z / num_el;

        // std::cout << "CPU results for " << parameters[0] << " " << parameters[1] << " " << parameters[2] << " ";
        // std::cout << residuals[0] << " " << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;
        // cudaMemset(image_,0,sizeof(double) * height_ * width_ * 4);
        // END SEMI CPU
        getContrastDelBatchReduce(image_, residuals, gradient, height_, width_,
                                  contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, means_, num_events_, stream_);

        // std::cout<<"reduction kernel "<<means[0]<<" "<<means[1]<<" "<<means[2]<<" "<<means[3]<<" "<<std::endl;
        // std::cout<<"results for "<<parameters[0]<<" "<<parameters[1]<<" "<<parameters[2]<<" ";
        // std::cout<<residuals[0]<<" "<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<std::endl;



        return true;
    }

    bool EvaluateCpu(const double *const parameters,
                     double *residuals,
                     double *gradient) const
    {
        nvtx3::scoped_range r{"Evaluate"};
        // fillImage(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, parameters[0], parameters[1], parameters[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);
        double x_unprojected[num_events_], y_unprojected[num_events_], t[num_events_];
        cudaMemcpy(x_unprojected, x_unprojected_, num_events_ * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(y_unprojected, y_unprojected_, num_events_ * sizeof(double), cudaMemcpyDefault);
        cudaMemcpy(t, t_, num_events_ * sizeof(double), cudaMemcpyDefault);

        double image[height_ * width_ * 4]={0};
        double *image_del_x = image + height_ * width_;
        double *image_del_y = image_del_x + height_ * width_;
        double *image_del_z = image_del_y + height_ * width_;


        for (int i = 0; i < num_events_; i++)
        {
            // calculate theta x,y,z
            double theta_x_t = parameters[0] * t[i];
            double theta_y_t = parameters[1] * t[i];
            double theta_z_t = parameters[2] * t[i];

            // calculate x/y/z_rotated
            double z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
            double x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
            double y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

            // calculate x_prime and y_prime
            double x_prime = fx_ * x_rotated_norm + cx_;
            double y_prime = fy_ * y_rotated_norm + cy_;
            // populate image
            int x_round = round(x_prime);
            int y_round = round(y_prime);
            double gaussian;

            if (x_round >= 1 && x_round <= width_ && y_round >= 1 && y_round <= height_)
            {
                double fx_div_z_rotated_ti = fx_ * z_rotated_inv * t[i];
                double fy_div_z_rotated_ti = fy_ * z_rotated_inv * t[i];
                double del_x_del_theta_y = fx_div_z_rotated_ti * (1 + x_unprojected[i] * x_rotated_norm);
                double del_x_del_theta_z = -fx_div_z_rotated_ti * y_unprojected[i];
                double del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
                double del_y_del_theta_x = fy_div_z_rotated_ti * (-1 - y_unprojected[i] * y_rotated_norm);
                double del_y_del_theta_z = fy_div_z_rotated_ti * x_unprojected[i];
                double del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;

                // for (int row = max(1,y_round - 3); row < min(height,y_round + 4); row++)
                // {
                //     for (int col = max(1,x_round - 3); col < min(width,x_round + 4); col++)
                //     {
                for (int row = std::max(1, y_round - 2); row < std::min(height_, y_round + 3); row++)
                {
                    for (int col = std::max(1, x_round - 2); col < std::min(width_, x_round + 3); col++)
                    {
                        double x_diff = col - x_prime;
                        double y_diff = row - y_prime;
                        // double x_diff = col - x_unprojected[i];
                        // double y_diff = row - y_unprojected[i];
                        // gaussian = exp((-x_diff * x_diff - y_diff * y_diff) / 2) / sqrt(2 * M_PI);
                        gaussian = exp((-x_diff * x_diff - y_diff * y_diff) / 2);
                        int idx = (row - 1) * (width_) + col - 1;
                        image[idx] += gaussian;
                        double del_x = gaussian * (x_diff * del_x_del_theta_x + y_diff * del_y_del_theta_x);
                        image_del_x[idx] += del_x;
                        double del_y = gaussian * (x_diff * del_x_del_theta_y + y_diff * del_y_del_theta_y);
                        image_del_y[idx] += del_y;
                        double del_z = gaussian * (x_diff * del_x_del_theta_z + y_diff * del_y_del_theta_z);
                        image_del_z[idx] += del_z;
                    }
                }
            }
        }

        double means[4]={0};
        // for (int i=0;i<4;i++){
        //     means[i]=thrustMean(image_+height_*width_*i,height_,width_);
        // }


        for (int i = 0; i < 4; i++)
        {
            for (int row = 0; row < height_; row++)
            {
                for (int col = 0; col < width_; col++)
                {
                    int idx = row * width_ + col;
                    means[i]+=image[idx+height_*width_*i];
                }
            }
                means[i]/=(height_*width_);
        }

        double image_contrast = 0;
        double image_contrast_del_theta_x = 0;
        double image_contrast_del_theta_y = 0;
        double image_contrast_del_theta_z = 0;
        // double image[height_ * width_ * 4];
        // cudaMemcpy(image, image_, sizeof(double) * height_ * width_ * 4, cudaMemcpyDefault);

        // cudaMemset(image_, 0, height_ * width_ * sizeof(double) * 4);
        for (int row = 0; row < height_; row++)
        {
            for (int col = 0; col < width_; col++)
            {
                int idx = row * width_ + col;
                double image_norm = image[idx] - means[0];
                double image_norm_x = image[idx + height_ * width_ * 1] - means[1];
                double image_norm_y = image[idx + height_ * width_ * 2] - means[2];
                double image_norm_z = image[idx + height_ * width_ * 3] - means[3];
                image_contrast += image_norm * image_norm;
                image_contrast_del_theta_x += image_norm_x * image_norm;
                image_contrast_del_theta_y += image_norm_y * image_norm;
                image_contrast_del_theta_z += image_norm_z * image_norm;
            }
        }

        int num_el = height_ * width_;
        residuals[0] = -image_contrast / num_el;
        gradient[0] = -2 * image_contrast_del_theta_x / num_el;
        gradient[1] = -2 * image_contrast_del_theta_y / num_el;
        gradient[2] = -2 * image_contrast_del_theta_z / num_el;

        std::cout << "CPU results for " << parameters[0] << " " << parameters[1] << " " << parameters[2] << " ";
        std::cout << residuals[0] << " " << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;

        return true;
    }
    int NumParameters() const override { return 3; }
    void SumImage()
    {
        std::cout << thrustMean(image_, height_, width_) << std::endl;
    }
    void GenerateImage(const double *const rotations, uint8_t *output_image, double &contrast)
    {
        double *image;
        warpEvents(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, rotations[0], rotations[1], rotations[2]);
        cudaMallocHost(&image, sizeof(double) * height_ * width_);
        std::fill_n(image, (height_) * (width_), 0);
        fillImageKronecker(height_, width_, num_events_, x_prime_, y_prime_, image);
        // fillImageKronecker(height_, width_, num_events_, x_, y_, image);
        cudaDeviceSynchronize();
        double maximum = getMax(image, height_, width_);
        // std::cout<<maximum<<std::endl;

        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
            }
        }
        cudaFreeHost(image);
    };
    void GenerateUncompensatedImage(const double *const rotations, uint8_t *output_image, double &contrast)
    {
        double *image;
        cudaMallocHost(&image, sizeof(double) * height_ * width_);
        std::fill_n(image, (height_) * (width_), 0);
        fillImageKronecker(height_, width_, num_events_, x_, y_, image);
        cudaDeviceSynchronize();
        double maximum = getMax(image, height_, width_);
        // std::cout<<"un max "<<maximum<<std::endl;
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
            }
        }
        cudaFreeHost(image);
    };
    void GenerateImageBilinear(const double *const rotations, uint8_t *output_image, double &contrast)
    {
        cudaMemsetAsync(image_, 0, height_ * width_ * sizeof(double));
        cudaDeviceSynchronize();
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, rotations[0], rotations[1], rotations[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);
        double *image;
        cudaMallocHost(&image, sizeof(double) * height_ * width_);
        cudaMemcpy(image, image_, height_ * width_ * sizeof(double), cudaMemcpyDefault);
        double maximum = getMax(image_, height_, width_);
        cudaMemsetAsync(image_, 0, height_ * width_ * sizeof(double));
        double mean = thrustMean(image_, height_, width_);
        // thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(image_);
        // double sum1 = thrust::reduce(dev_ptr, dev_ptr+height_*width_, 0.0, thrust::plus<double>());
        // double mean= sum1/(height_*width_);

        double contrast_sum = 0;
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
                contrast_sum += (image[(i) * (width_) + j] - mean) * (image[(i) * (width_) + j] - mean);
            }
        }
        contrast = contrast_sum / (height_ * width_);
        cudaFreeHost(image);
    };
    void GenerateUncompensatedImageBilinear(const double *const rotations, uint8_t *output_image, double &contrast)
    {
        cudaDeviceSynchronize();
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_, 0, 0, 0, contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);
        double *image;
        cudaMallocHost(&image, sizeof(double) * height_ * width_);
        cudaMemcpy(image, image_, height_ * width_ * sizeof(double), cudaMemcpyDefault);
        double maximum = getMax(image_, height_, width_);
        cudaMemsetAsync(image_, 0, height_ * width_ * sizeof(double));
        double mean = thrustMean(image_, height_, width_);
        double contrast_sum = 0;
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
                contrast_sum += (image[(i) * (width_) + j] - mean) * (image[(i) * (width_) + j] - mean);
            }
        }
        contrast = contrast_sum / (height_ * width_);
        cudaFreeHost(image);
    };

    int f_count = 0;
    int g_count = 0;

    // private:
    thrust::device_vector<double> H;
    double *x_unprojected_ = NULL;
    double *y_unprojected_ = NULL;
    double *x_ = NULL;
    double *y_ = NULL;
    double *x_prime_ = NULL;
    double *y_prime_ = NULL;
    double *t_ = NULL;
    int height_;
    int width_;
    int num_events_;
    int max_num_events_ = 30000;
    double *image_ = NULL;
    // double *image_del_theta_x_ = NULL;
    // double *image_del_theta_y_ = NULL;
    // double *image_del_theta_z_ = NULL;
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    double *contrast_block_sum_;
    double *contrast_del_x_block_sum_;
    double *contrast_del_y_block_sum_;
    double *contrast_del_z_block_sum_;
    double *means_;
    std::shared_ptr<std::thread> memset_thread_;
    std::mutex m_;
    std::shared_ptr<std::condition_variable> cv_;
    bool running = true;
    cudaStream_t stream_[2];
    bool allocated_ = false;
};

class McGradientInterface final : public ceres::FirstOrderFunction
{
public:
    McGradientInterface(McGradient *mc_gr) : mc_gr_(mc_gr)
    {
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const override
    {
        return mc_gr_->Evaluate(parameters, residuals, gradient);
    }
    int NumParameters() const override { return 3; }

private:
    McGradient *mc_gr_;
};
#endif // MC_GRADIENT_H