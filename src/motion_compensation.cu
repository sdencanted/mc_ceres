#include "motion_compensation.h"

#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <vector>
#include <iostream>
#include "utils.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include <jetson-utils/cudaMappedMemory.h>

__global__ void fillImage_(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z, bool do_jacobian, double *image_del_x, double *image_del_y, double *image_del_z)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        double gaussian;
        // calculate theta x,y,z
        double theta_x_t = rotation_x * t[i];
        double theta_y_t = rotation_y * t[i];
        double theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        double z_rotated_inv = 1/(-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        double x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        double y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + cx;
        y_prime[i] = fy * y_rotated_norm + cy;

        double del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
        if (do_jacobian)
        {
            float fx_div_z_rotated=fx*z_rotated_inv;
            float fy_div_z_rotated=fy*z_rotated_inv;
            del_x_del_theta_y = fx_div_z_rotated * t[i] * (1 + x_unprojected[i] * x_rotated_norm);
            del_x_del_theta_z = fx_div_z_rotated * (-t[i] * y_unprojected[i]);
            del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
            del_y_del_theta_x = fy_div_z_rotated * t[i] * (-1 - y_unprojected[i] * y_rotated_norm);
            del_y_del_theta_z = fy_div_z_rotated * (t[i] * x_unprojected[i]);
            del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;
        }
        // populate image
        // check if coordinates are 3 pixels in of the boundary
        int x_round = round(x_prime[i]);
        int y_round = round(y_prime[i]);
        // int x_round = round(x_unprojected[i]);
        // int y_round = round(y_unprojected[i]);
        if (x_round >= 1 && x_round <= width && y_round >= 1 && y_round <= height)
        {
            for (int row = y_round - 3; row < y_round + 4; row++)
            {
                for (int col = x_round - 3; col < x_round + 4; col++)
                {
                    // TODO: make a LUT for the values here rounded to a certain s.f. and see if there is a speed-up
                    double x_diff = col - x_prime[i];
                    double y_diff = row - y_prime[i];
                    // double x_diff = col - x_unprojected[i];
                    // double y_diff = row - y_unprojected[i];
                    gaussian = exp((-x_diff * x_diff - y_diff * y_diff) / 2) / sqrt(2 * M_PI);
                    int idx = (row + 3 - 1) * (width + 6) + col + 3 - 1;
                    atomicAdd(&image[idx], gaussian);
                    if (do_jacobian)
                    {
                        atomicAdd(&image_del_x[idx], gaussian * (x_diff * del_x_del_theta_x + y_diff * del_y_del_theta_x));
                        atomicAdd(&image_del_y[idx], gaussian * (x_diff * del_x_del_theta_y + y_diff * del_y_del_theta_y));
                        atomicAdd(&image_del_z[idx], gaussian * (x_diff * del_x_del_theta_z + y_diff * del_y_del_theta_z));
                    }
                }
            }
        }
    }
}
void fillImage(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z, bool do_jacobian, double *image_del_x, double *image_del_y, double *image_del_z)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    fillImage_<<<blocks_per_sm * num_sm, threads_per_block>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, do_jacobian, image_del_x, image_del_y, image_del_z);
}

__global__ void fillImageKronecker_(int height, int width, int num_events, double *x_prime, double *y_prime, double *image)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        // populate image
        // check if coordinates are 3 pixels in of the boundary
        int x_round = round(x_prime[i]);
        int y_round = round(y_prime[i]);
        if (x_round >= 1 && x_round <= width && y_round >= 1 && y_round <= height)
        {
            int idx = (y_round + 3 - 1) * (width + 6) + x_round + 3 - 1;
            atomicAdd(&image[idx], 1);
        }
    }
}
void fillImageKronecker(int height, int width, int num_events, double *x_prime, double *y_prime, double *image)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    cudaMemset(image, 0, height * width * sizeof(double));
    fillImageKronecker_<<<blocks_per_sm * num_sm, threads_per_block>>>(height, width, num_events, x_prime, y_prime, image);
}
double getMeanCustom(double *image, int height, int width)
{
    return gpu_sum_reduce(image, (height + 6) * (width + 6)) / height * width;
}
int getCubSize(double *image, int height, int width)
{
    size_t temp_cub_temp_size;
    double *temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, image, (height + 6) * (width + 6), cub::Sum(), 0);
    return temp_cub_temp_size;
}
double getMean(double *image, int height, int width, int cub_temp_size)
{
    // return gpu_sum_reduce(image, height * width) / height * width;
    // double *in;
    // cudaMalloc(&in, sizeof(double) * (height + 6) * (width + 6));
    // cudaMemset(in, (double)1.0, sizeof(double) * (height + 6) * (width + 6));
    double *out;
    cudaMalloc(&out, sizeof(double));
    // std::cout << "allocating cuda mem for out" << std::endl;
    // if (!cudaAllocMapped((void**)&out, sizeof(double)))
    // {
    //     std::cout << "could not allocate cuda mem for out" << std::endl;
    // }
    size_t temp_cub_temp_size;
    double *temp_storage = NULL;
    if (cub_temp_size != 0)
    {
        temp_cub_temp_size = cub_temp_size;
    }
    else
    {
        std::cout<<"no cub temp size"<<std::endl;
        cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height + 6) * (width + 6), cub::Sum(), 0);
        cudaDeviceSynchronize();
    }
    cudaMalloc(&temp_storage, temp_cub_temp_size);
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height + 6) * (width + 6), cub::Sum(), 0);
    cudaDeviceSynchronize();
    double sum;
    cudaMemcpy(&sum, out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(out);
    return sum / ((height + 6) * (width + 6));
}
double getMeanCpu(double *image, int height, int width)
{

    double sum = 0;
    for (int i = 0; i < height + 6; i++)
    {
        for (int j = 0; j < width + 6; j++)
            sum += image[i * (width + 6) + j];
    }
    return sum / ((height + 6) * (width + 6));
}
double getMeanCpuCrop(double *image, int height, int width)
{

    double sum = 0;
    for (int i = 3; i < height + 3; i++)
    {
        for (int j = 3; j < width + 3; j++)
        {
            sum += image[i * (width + 6) + j];
        }
    }
    return sum / (height * width);
}
__global__ void subtractMean_(double *image, int height, int width, double mean)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for (size_t i = thread_grid_idx; i < (height + 6) * (width + 6); i += num_threads_in_grid)
    {
        atomicAdd(&image[i], -mean);
    }
}
void subtractMean(double *image, int height, int width, int cub_temp_size)
{
    // double mean = getMeanCpuCrop(image, height, width);
    // double mean_crop = getMeanCpuCrop(image, height, width);
    // double mean_custom = getMeanCustom(image, height+6, width+6);
    double mean_cuda = getMean(image, height, width, cub_temp_size);
    // std::cout << "subtracting " << mean_cuda  <<" "<<mean_custom<< std::endl;
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    subtractMean_<<<blocks_per_sm * num_sm, threads_per_block>>>(image, height, width, mean_cuda);
}
__global__ void getContrast_(double *image, int height, int width, double *image_out)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // for (size_t i = thread_grid_idx; i < (height) * (width + 6); i += num_threads_in_grid)
    for (size_t i = thread_grid_idx; i < (height + 6) * (width + 6); i += num_threads_in_grid)
    {
        image_out[i]=image[i] * image[i];
        // atomicAdd(contrast, image[i] * image[i]);
        // if (i % (width + 6) >= 3 && i % (width + 6) < width + 3)
        // {
        //     atomicAdd(contrast, image[i + 3 * (width + 6)] * image[i + 3 * (width + 6)]);
        // }
    }
}
double getContrast(double *image, int height, int width, int cub_temp_size)
{
    double contrast = 0;

    double *temp_image;
    checkCudaErrors(cudaMalloc((void **)&temp_image, (unsigned int)sizeof(double)*(height+6)*(width+6)));
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    getContrast_<<<blocks_per_sm * num_sm, threads_per_block>>>(image, height, width, temp_image);
    contrast=getMean(temp_image,height,width,cub_temp_size);
    checkCudaErrors(cudaFree(temp_image));
    return contrast;
}

__global__ void getContrastDel_(double *image, double *image_del, int height, int width)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // for (size_t i = thread_grid_idx; i < (height) * (width + 6); i += num_threads_in_grid)
    for (size_t i = thread_grid_idx; i < (height + 6) * (width + 6); i += num_threads_in_grid)
    {
        image_del[i] *= image[i];
        // atomicAdd(contrast, image[i] * image_del[i]);
    }
}
double getContrastDel(double *image, double *image_del, int height, int width, int cub_temp_size)
{
    double contrast = 0;
    // double *contrast_device;
    // checkCudaErrors(cudaMalloc((void **)&contrast_device, (unsigned int)sizeof(double)));
    // checkCudaErrors(cudaMemcpy(contrast_device, &contrast, sizeof(double), cudaMemcpyHostToDevice));
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    getContrastDel_<<<blocks_per_sm * num_sm, threads_per_block>>>(image, image_del, height, width);
    contrast = 2 * getMean(image_del, height, width,cub_temp_size);
    // checkCudaErrors(cudaMemcpy(&contrast, contrast_device, sizeof(double), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaFree(contrast_device));
    return contrast;
    // return 2 * contrast / (height * width);
    // return 2 * contrast / ((height + 6) * (width + 6));
}