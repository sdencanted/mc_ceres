#include "motion_compensation_double.h"

#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <vector>
#include <iostream>
#include "utils.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include <jetson-utils/cudaMappedMemory.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#define FULL_MASK 0xffffffff
// using namespace cooperative_groups;
double thrustMean(double *image_, int height_, int width_)
{
    thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(image_);
    double sum1 = thrust::reduce(dev_ptr, dev_ptr + height_ * width_, 0.0, thrust::plus<double>());
    return sum1 / (height_ * width_);
}
double thrustSum(double *image_, int num_el)
{
    thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(image_);
    double sum1 = thrust::reduce(dev_ptr, dev_ptr + num_el, 0.0, thrust::plus<double>());
    return sum1;
}
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

__global__ void fillImage_(double fx, double fy, double cx, double cy, int height, int width, int num_events, const double *x_unprojected, const double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum)
{

    double image_sum = 0;
    double image_sum_del_theta_x = 0;
    double image_sum_del_theta_y = 0;
    double image_sum_del_theta_z = 0;
    double *image_del_x = image + height * width;
    double *image_del_y = image + height * width * 2;
    double *image_del_z = image + height * width * 3;
    // size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // if (i < num_events)
    for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x); i < num_events; i += num_threads_in_grid)
    {
        // calculate theta x,y,z
        double theta_x_t = rotation_x * t[i];
        double theta_y_t = rotation_y * t[i];
        double theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        double z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        double x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        double y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + cx;
        y_prime[i] = fy * y_rotated_norm + cy;
        // populate image
        int x_round = round(x_prime[i]);
        int y_round = round(y_prime[i]);
        double gaussian;

        if (x_round >= 1 && x_round <= width && y_round >= 1 && y_round <= height)
        {
            double fx_div_z_rotated_ti = fx * z_rotated_inv * t[i];
            double fy_div_z_rotated_ti = fy * z_rotated_inv * t[i];
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
            for (int row = max(1, y_round - 2); row < min(height, y_round + 3); row++)
            {
                for (int col = max(1, x_round - 2); col < min(width, x_round + 3); col++)
                {
                    // TODO: make a LUT for the values here rounded to a certain s.f. and see if there is a speed-up
                    double x_diff = col - x_prime[i];
                    double y_diff = row - y_prime[i];
                    // double x_diff = col - x_unprojected[i];
                    // double y_diff = row - y_unprojected[i];
                    // gaussian = exp((-x_diff * x_diff - y_diff * y_diff) / 2) / sqrt(2 * M_PI);
                    gaussian = exp((-x_diff * x_diff - y_diff * y_diff) / 2);
                    int idx = (row - 1) * (width) + col - 1;
                    atomicAdd(&image[idx], gaussian);
                    image_sum += gaussian;
                    double del_x = gaussian * (x_diff * del_x_del_theta_x + y_diff * del_y_del_theta_x);
                    atomicAdd(&image_del_x[idx], del_x);
                    image_sum_del_theta_x += del_x;
                    double del_y = gaussian * (x_diff * del_x_del_theta_y + y_diff * del_y_del_theta_y);
                    atomicAdd(&image_del_y[idx], del_y);
                    image_sum_del_theta_y += del_y;
                    double del_z = gaussian * (x_diff * del_x_del_theta_z + y_diff * del_y_del_theta_z);
                    atomicAdd(&image_del_z[idx], del_z);
                    image_sum_del_theta_z += del_z;
                }
            }
        }
    }
    double *sdata = SharedMemory<double>();
    uint16_t tid = threadIdx.x;

    // do reduction in shared mem

    // sum up to 128 elements

    double temp_sum;
    // image_sum
    sdata[tid] = image_sum;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum = image_sum + sdata[tid + 256];
    __syncthreads();
    // store contrast in 0 to 127
    if (tid < 128)
        temp_sum = image_sum + sdata[tid + 128];
    __syncthreads();
    // image_sum_del_theta_x
    sdata[tid] = image_sum_del_theta_x;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 128];
    __syncthreads();
    // store x in 128 to 255
    if (tid >= 128 && tid < 256)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_sum_del_theta_y
    sdata[tid] = image_sum_del_theta_y;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 128];
    __syncthreads();
    // store y in 256 to 383
    if (tid >= 256 && tid < 384)
    {
        temp_sum = sdata[tid - 256];
    }
    __syncthreads();
    // image_sum_del_theta_z
    sdata[tid] = image_sum_del_theta_z;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
    {
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 128];
    }
    __syncthreads();
    // store z in 384 to 512
    if (tid >= 384)
    {
        temp_sum = sdata[tid - 384];
    }
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x7F) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid & 0x7F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        // image_sum
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_sum_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 256)
    {
        // image_sum_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 384)
    {
        // image_sum_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}

__global__ void warpEvents_(double fx, double fy, double cx, double cy, int height, int width, int num_events, const double *x_unprojected, const double *y_unprojected, double *x_prime, double *y_prime, double *t, const double rotation_x, const double rotation_y, const double rotation_z)
{
    // size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // if (i < num_events)
    for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x); i < num_events; i += num_threads_in_grid)
    {
        // calculate theta x,y,z
        double theta_x_t = rotation_x * t[i];
        double theta_y_t = rotation_y * t[i];
        double theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        double z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        double x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        double y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + cx;
        y_prime[i] = fy * y_rotated_norm + cy;
    }
}
void warpEvents(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, const double rotation_x, const double rotation_y, const double rotation_z)
{
    int blockSize = 512; // The launch configurator returned block size
    int gridSize;        // The actual grid size needed, based on input size
    gridSize = (num_events + blockSize - 1) / blockSize;
    int smemSize = blockSize * sizeof(double);
    warpEvents_<<<gridSize, blockSize, smemSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, rotation_x, rotation_y, rotation_z);
}
__global__ void fillImageBilinear_(double fx, double fy, double cx, double cy, int height, int width, int num_events, const double *x_unprojected, const double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum)
{

    double image_sum = 0;
    double image_sum_del_theta_x = 0;
    double image_sum_del_theta_y = 0;
    double image_sum_del_theta_z = 0;
    double *image_del_x = image + height * width;
    double *image_del_y = image + height * width * 2;
    double *image_del_z = image + height * width * 3;
    // size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // if (i < num_events)
    for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x); i < num_events; i += num_threads_in_grid)
    {
        // calculate theta x,y,z
        double theta_x_t = rotation_x * t[i];
        double theta_y_t = rotation_y * t[i];
        double theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        double z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        double x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        double y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + cx;
        y_prime[i] = fy * y_rotated_norm + cy;
        // populate image

        // Bilinear
        int x_trunc = int(x_prime[i]);
        int y_trunc = int(y_prime[i]);
        if (x_trunc >= 1 && x_trunc <= width - 2 && y_trunc >= 1 && y_trunc <= height - 2)
        {

            // int idx1 = x_trunc - 1 + (y_trunc - 1) * width;
            // int idx2 = idx1 + 1;
            // int idx3 = idx1 + width;
            // int idx4 = idx3 + 1;

            int idx4 = x_trunc + y_trunc * width;
            int idx3 = idx4 - 1;
            int idx2 = idx4 - width;
            int idx1 = idx2 - 1;
            double x_diff = x_prime[i] - x_trunc;
            double y_diff = y_prime[i] - y_trunc;
            double del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
            double fx_div_z_rotated_ti = fx * z_rotated_inv * t[i];
            double fy_div_z_rotated_ti = fy * z_rotated_inv * t[i];
            del_x_del_theta_y = fx_div_z_rotated_ti * (1 + x_unprojected[i] * x_rotated_norm);
            del_x_del_theta_z = fx_div_z_rotated_ti * -y_unprojected[i];
            del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
            del_y_del_theta_x = fy_div_z_rotated_ti * (-1 - y_unprojected[i] * y_rotated_norm);
            del_y_del_theta_z = fy_div_z_rotated_ti * x_unprojected[i];
            del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;
            // double d1x = -(1 - y_diff);
            // double d1y = -(1 - x_diff);
            double d2x = 1 - y_diff;
            double d2y = -x_diff;
            double d3x = -y_diff;
            double d3y = 1 - x_diff;
            double d4x = y_diff;
            double d4y = x_diff;

            double d1x = -d2x;
            double d1y = -d3y;

            // double im1 = (1 - x_diff) * (1 - y_diff);
            double im1 = d3y * d2x;
            // double im2 = (x_diff) * (1 - y_diff);
            double im2 = d4y * d2x;
            // double im3 = (1 - x_diff) * (y_diff);
            double im3 = d3y * y_diff;
            double im4 = (x_diff) * (y_diff);
            image_sum = im1 + im2 + im3 + im4;
            atomicAdd(&image[idx1], im1);
            atomicAdd(&image[idx2], im2);
            atomicAdd(&image[idx3], im3);
            atomicAdd(&image[idx4], im4);
            double dx1 = d1x * del_x_del_theta_x + d1y * del_y_del_theta_x;
            double dx2 = d2x * del_x_del_theta_x + d2y * del_y_del_theta_x;
            double dx3 = d3x * del_x_del_theta_x + d3y * del_y_del_theta_x;
            double dx4 = d4x * del_x_del_theta_x + d4y * del_y_del_theta_x;
            image_sum_del_theta_x = dx1 + dx2 + dx3 + dx4;

            atomicAdd(&image_del_x[idx1], dx1);
            atomicAdd(&image_del_x[idx2], dx2);
            atomicAdd(&image_del_x[idx3], dx3);
            atomicAdd(&image_del_x[idx4], dx4);
            double dy1 = d1x * del_x_del_theta_y + d1y * del_y_del_theta_y;
            double dy2 = d2x * del_x_del_theta_y + d2y * del_y_del_theta_y;
            double dy3 = d3x * del_x_del_theta_y + d3y * del_y_del_theta_y;
            double dy4 = d4x * del_x_del_theta_y + d4y * del_y_del_theta_y;
            image_sum_del_theta_y = dy1 + dy2 + dy3 + dy4;
            atomicAdd(&image_del_y[idx1], dy1);
            atomicAdd(&image_del_y[idx2], dy2);
            atomicAdd(&image_del_y[idx3], dy3);
            atomicAdd(&image_del_y[idx4], dy4);
            double dz1 = d1x * del_x_del_theta_z + d1y * del_y_del_theta_z;
            double dz2 = d2x * del_x_del_theta_z + d2y * del_y_del_theta_z;
            double dz3 = d3x * del_x_del_theta_z + d3y * del_y_del_theta_z;
            double dz4 = d4x * del_x_del_theta_z + d4y * del_y_del_theta_z;
            image_sum_del_theta_z = dz1 + dz2 + dz3 + dz4;
            atomicAdd(&image_del_z[idx1], dz1);
            atomicAdd(&image_del_z[idx2], dz2);
            atomicAdd(&image_del_z[idx3], dz3);
            atomicAdd(&image_del_z[idx4], dz4);
        }
    }
    double *sdata = SharedMemory<double>();
    uint16_t tid = threadIdx.x;

    // do reduction in shared mem

    // sum up to 128 elements

    double temp_sum;
    // image_sum
    sdata[tid] = image_sum;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum = image_sum + sdata[tid + 256];
    __syncthreads();
    // store contrast in 0 to 127
    if (tid < 128)
        temp_sum = image_sum + sdata[tid + 128];
    __syncthreads();
    // image_sum_del_theta_x
    sdata[tid] = image_sum_del_theta_x;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 128];
    __syncthreads();
    // store x in 128 to 255
    if (tid >= 128 && tid < 256)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_sum_del_theta_y
    sdata[tid] = image_sum_del_theta_y;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 128];
    __syncthreads();
    // store y in 256 to 383
    if (tid >= 256 && tid < 384)
    {
        temp_sum = sdata[tid - 256];
    }
    __syncthreads();
    // image_sum_del_theta_z
    sdata[tid] = image_sum_del_theta_z;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
    {
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 128];
    }
    __syncthreads();
    // store z in 384 to 512
    if (tid >= 384)
    {
        temp_sum = sdata[tid - 384];
    }
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x7F) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid & 0x7F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        // image_sum
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_sum_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 256)
    {
        // image_sum_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 384)
    {
        // image_sum_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}
void fillImageBilinear(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum)
{
    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize = 512; // The launch configurator returned block size
    // int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
    //                                    fillImageBilinear_, 0, 0);
    // Round up according to array size
    gridSize = std::min(128, (num_events + blockSize - 1) / blockSize);

    int smemSize = blockSize * sizeof(double);
    fillImageBilinear_<<<gridSize, blockSize, smemSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
}
void fillImage(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum)
{
    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize = 512; // The launch configurator returned block size
    // int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
    //                                    fillImageBilinear_, 0, 0);
    // Round up according to array size
    gridSize = std::min(128, (num_events + blockSize - 1) / blockSize);

    int smemSize = blockSize * sizeof(double);
    fillImage_<<<gridSize, blockSize, smemSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
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
            int idx = (y_round - 1) * width + x_round - 1;
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
double getMax(double *image, int height, int width)
{
    double *out;
    cudaMalloc(&out, sizeof(double));
    size_t temp_cub_temp_size;
    double *temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height) * (width), cub::Max(), 0);
    cudaDeviceSynchronize();
    cudaMalloc(&temp_storage, temp_cub_temp_size);
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height) * (width), cub::Max(), 0);
    cudaDeviceSynchronize();
    double maximum;
    cudaMemcpy(&maximum, out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(out);
    cudaFree(temp_storage);
    return maximum;
}
__device__ volatile double mean_volatile[4] = {0};
// __global__ void getContrastDelBatchReduceHarder256_(double *image, int num_elements, double *means, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum, int prev_gridsize, double *imagedebug)
__global__ void getContrastDelBatchReduceHarder256_(double *image, int num_elements, double *means, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum, int prev_gridsize)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    double *image_del_x = image + num_elements;
    double *image_del_y = image + num_elements * 2;
    double *image_del_z = image + num_elements * 3;
    // START COPY
    double *sdata = SharedMemory<double>();
    double temp_sum = 0;
    uint16_t tid = threadIdx.x;
    // 85 partial sums to go
    // dump partial sums inside again
    if (tid < prev_gridsize)
    {

        if (blockIdx.x == 0)
        {
            temp_sum = contrast_block_sum[tid];
        }
        else if (blockIdx.x == 1)
        {
            temp_sum = contrast_del_x_block_sum[tid];
        }
        else if (blockIdx.x == 2)
        {
            temp_sum = contrast_del_y_block_sum[tid];
        }
        else if (blockIdx.x == 3)
        {
            temp_sum = contrast_del_z_block_sum[tid];
        }
    }
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    if (tid == 0)
    {
        // write means
        if (blockIdx.x < 4)
        {
            mean_volatile[blockIdx.x] = temp_sum / num_elements;
        }
    }
    // __syncthreads();

    // END COPY
    double image_contrast = 0;
    double image_contrast_del_theta_x = 0;
    double image_contrast_del_theta_y = 0;
    double image_contrast_del_theta_z = 0;
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t idx = thread_grid_idx;
    cooperative_groups::sync(grid);
    double mean = mean_volatile[0];
    double xmean = mean_volatile[1];
    double ymean = mean_volatile[2];
    double zmean = mean_volatile[3];

    while (idx < num_elements)
    {
        double image_norm = image[idx] - mean;
        double image_norm_x = image_del_x[idx] - xmean;
        double image_norm_y = image_del_y[idx] - ymean;
        double image_norm_z = image_del_z[idx] - zmean;
        image_contrast += image_norm * image_norm;
        image_contrast_del_theta_x += image_norm_x * image_norm;
        image_contrast_del_theta_y += image_norm_y * image_norm;
        image_contrast_del_theta_z += image_norm_z * image_norm;
        // imagedebug[idx] = image_norm * image_norm;
        // imagedebug[idx + num_elements] = image_norm_x * image_norm;
        // imagedebug[idx + num_elements * 2] = image_norm_y * image_norm;
        // imagedebug[idx + num_elements * 3] = image_norm_z * image_norm;
        idx += blockDim.x * gridDim.x;
    }
    // BEGIN DEBUG
    // sdata[tid] = 0;
    // if(blockIdx.x==0&&threadIdx.x==0){
    //     for(int idx_special=0;idx_special<num_elements;idx_special++)

    //     {
    //         double image_norm = image[idx_special] - mean;
    //         double image_norm_x = image_del_x[idx_special] - xmean;
    //         double image_norm_y = image_del_y[idx_special] - ymean;
    //         double image_norm_z = image_del_z[idx_special] - zmean;
    //         // image_contrast += image_norm * image_norm;
    //         // image_contrast_del_theta_x += image_norm_x * image_norm;
    //         // image_contrast_del_theta_y += image_norm_y * image_norm;
    //         // image_contrast_del_theta_z += image_norm_z * image_norm;
    //         // imagedebug[idx_special] = image_norm * image_norm;
    //         // imagedebug[idx_special + num_elements] = image_norm_x * image_norm;
    //         // imagedebug[idx_special + num_elements * 2] = image_norm_y * image_norm;
    //         // imagedebug[idx_special + num_elements * 3] = image_norm_z * image_norm;

    //         atomicAdd(&sdata[0], image_norm * image_norm);
    //         atomicAdd(&sdata[1], image_norm_x * image_norm);
    //         atomicAdd(&sdata[2], image_norm_y * image_norm);
    //         atomicAdd(&sdata[3], image_norm_z * image_norm);

    //     }

    //     contrast_block_sum[blockIdx.x] = sdata[0];
    //     contrast_del_x_block_sum[blockIdx.x] = sdata[1];
    //     contrast_del_y_block_sum[blockIdx.x] = sdata[2];
    //     contrast_del_z_block_sum[blockIdx.x] = sdata[3];
    //     // contrast_block_sum[blockIdx.x] = image_contrast;
    //     // contrast_del_x_block_sum[blockIdx.x] = image_contrast_del_theta_x;
    //     // contrast_del_y_block_sum[blockIdx.x] = image_contrast_del_theta_y;
    //     // contrast_del_z_block_sum[blockIdx.x] = image_contrast_del_theta_z;
    // }
    // else{

    //     contrast_block_sum[blockIdx.x] =0;
    //     contrast_del_x_block_sum[blockIdx.x] =0;
    //     contrast_del_y_block_sum[blockIdx.x] =0;
    //     contrast_del_z_block_sum[blockIdx.x] = 0;
    // }
    // return;
    // END DEBUG

    // do reduction in shared mem

    // sum up to 128 elements

    // double temp_sum;
    // image_contrast
    sdata[tid] = image_contrast;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast = image_contrast + sdata[tid + 128];
    __syncthreads();
    // store contrast in 0 to 63
    if (tid < 64)
        temp_sum = image_contrast + sdata[tid + 64];
    __syncthreads();
    // image_contrast_del_theta_x
    sdata[tid] = image_contrast_del_theta_x;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_x = image_contrast_del_theta_x + sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] = image_contrast_del_theta_x = image_contrast_del_theta_x + sdata[tid + 64];
    __syncthreads();
    // store x in 64 to 127
    if (tid >= 64 && tid < 128)
    {
        temp_sum = sdata[tid - 64];
    }
    __syncthreads();
    // image_contrast_del_theta_y
    sdata[tid] = image_contrast_del_theta_y;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] = image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 64];
    __syncthreads();
    // store y in 128 to 191
    if (tid >= 128 && tid < 192)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_contrast_del_theta_z
    sdata[tid] = image_contrast_del_theta_z;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_z = image_contrast_del_theta_z + sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
    {
        sdata[tid] = image_contrast_del_theta_z = image_contrast_del_theta_z + sdata[tid + 64];
    }
    __syncthreads();
    // store z in 192 to 255
    if (tid >= 192)
    {
        temp_sum = sdata[tid - 192];
    }
    __syncthreads();
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x3F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }

    if (tid == 0)
    {
        // image_contrast
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 64)
    {
        // image_contrast_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_contrast_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 192)
    {
        // image_contrast_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}
__global__ void getContrastDelBatchReduceHarder_(double *image, int num_elements, double *means, double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum, int prev_gridsize)
{
    double *image_del_x = image + num_elements;
    double *image_del_y = image + num_elements * 2;
    double *image_del_z = image + num_elements * 3;
    // START COPY
    double *sdata = SharedMemory<double>();
    double temp_sum = 0;
    uint16_t tid = threadIdx.x;
    // 85 partial sums to go
    // dump partial sums inside again
    if (tid < prev_gridsize)
    {

        if (blockIdx.x == 0)
        {
            temp_sum = contrast_block_sum[tid];
        }
        else if (blockIdx.x == 1)
        {
            temp_sum = contrast_del_x_block_sum[tid];
        }
        else if (blockIdx.x == 2)
        {
            temp_sum = contrast_del_y_block_sum[tid];
        }
        else if (blockIdx.x == 3)
        {
            temp_sum = contrast_del_z_block_sum[tid];
        }
    }
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    if (tid == 0)
    {
        if (blockIdx.x == 0)
        {
            means[0] = temp_sum / num_elements;
        }
        else if (blockIdx.x == 1)
        {
            means[1] = temp_sum / num_elements;
        }
        else if (blockIdx.x == 2)
        {
            means[2] = temp_sum / num_elements;
        }
        else if (blockIdx.x == 3)
        {
            means[3] = temp_sum / num_elements;
        }
    }

    // END COPY
    double image_contrast = 0;
    double image_contrast_del_theta_x = 0;
    double image_contrast_del_theta_y = 0;
    double image_contrast_del_theta_z = 0;
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    // size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    size_t idx = thread_grid_idx;
    // __syncthreads();
    auto g = cooperative_groups::this_grid();
    // auto g = cooperative_groups::this_thread_block();
    g.sync();
    double mean = means[0];
    double xmean = means[1];
    double ymean = means[2];
    double zmean = means[3];

    while (idx < num_elements)
    {
        double image_norm = image[idx] - mean;
        double image_norm_x = image_del_x[idx] - xmean;
        double image_norm_y = image_del_y[idx] - ymean;
        double image_norm_z = image_del_z[idx] - zmean;
        image_contrast += image_norm * image_norm;
        image_contrast_del_theta_x += image_norm_x * image_norm;
        image_contrast_del_theta_y += image_norm_y * image_norm;
        image_contrast_del_theta_z += image_norm_z * image_norm;
        idx += blockDim.x * gridDim.x;
    }

    // do reduction in shared mem

    // sum up to 128 elements

    // double temp_sum;
    // image_contrast
    sdata[tid] = image_contrast;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_contrast = image_contrast + sdata[tid + 256];
    __syncthreads();
    // store contrast in 0 to 127
    if (tid < 128)
        temp_sum = image_contrast + sdata[tid + 128];
    __syncthreads();
    // image_contrast_del_theta_x
    sdata[tid] = image_contrast_del_theta_x;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_contrast_del_theta_x = image_contrast_del_theta_x + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_x = image_contrast_del_theta_x + sdata[tid + 128];
    __syncthreads();
    // store x in 128 to 255
    if (tid >= 128 && tid < 256)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_contrast_del_theta_y
    sdata[tid] = image_contrast_del_theta_y;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 128];
    __syncthreads();
    // store y in 256 to 383
    if (tid >= 256 && tid < 384)
    {
        temp_sum = sdata[tid - 256];
    }
    __syncthreads();
    // image_contrast_del_theta_z
    sdata[tid] = image_contrast_del_theta_z;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_contrast_del_theta_z = image_contrast_del_theta_z + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
    {
        sdata[tid] = image_contrast_del_theta_z = image_contrast_del_theta_z + sdata[tid + 128];
    }
    __syncthreads();
    // store z in 384 to 512
    if (tid >= 384)
    {
        temp_sum = sdata[tid - 384];
    }
    __syncthreads();
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x7F) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid & 0x7F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }

    if (tid == 0)
    {
        // image_contrast
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_contrast_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 256)
    {
        // image_contrast_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 384)
    {
        // image_contrast_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}

// 4 blocks x threads
template <int prev_gridsize>
__global__ void getContrastDelBatchReduceHarderPt2_(double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum)
{
    double *sdata = SharedMemory<double>();

    double temp_sum = 0;
    uint16_t tid = threadIdx.x;
    // 85 partial sums to go
    // dump partial sums inside again
    if (tid < prev_gridsize)
    {

        if (blockIdx.x == 0)
        {
            temp_sum = temp_sum = contrast_block_sum[tid];
        }
        else if (blockIdx.x == 1)
        {
            temp_sum = contrast_del_x_block_sum[tid];
        }
        else if (blockIdx.x == 2)
        {
            temp_sum = contrast_del_y_block_sum[tid];
        }
        else if (blockIdx.x == 3)
        {
            temp_sum = contrast_del_z_block_sum[tid];
        }
    }
    sdata[tid] = temp_sum;
    __syncthreads();

    if (prev_gridsize > 256 && (tid) < 256)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 256];
    }
    __syncthreads();
    if (prev_gridsize > 128 && (tid) < 128)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 128];
    }
    __syncthreads();
    if (prev_gridsize > 64 && (tid) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    if (tid == 0)
    {
        if (blockIdx.x == 0)
        {
            contrast_block_sum[0] = temp_sum;
        }
        else if (blockIdx.x == 1)
        {
            contrast_block_sum[1] = temp_sum;
        }
        else if (blockIdx.x == 2)
        {
            contrast_block_sum[2] = temp_sum;
        }
        else
        {
            contrast_block_sum[3] = temp_sum;
        }
    }
}

// 4 blocks 128 threads
__global__ void meanPt2_(double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum, int num_elements, double *means, int prev_gridsize)
{
    double *sdata = SharedMemory<double>();
    double temp_sum = 0;
    uint16_t tid = threadIdx.x;
    // 85 partial sums to go
    // dump partial sums inside again
    if (tid < prev_gridsize)
    {

        if (blockIdx.x == 0)
        {
            temp_sum = contrast_block_sum[tid];
        }
        else if (blockIdx.x == 1)
        {
            temp_sum = contrast_del_x_block_sum[tid];
        }
        else if (blockIdx.x == 2)
        {
            temp_sum = contrast_del_y_block_sum[tid];
        }
        else
        {
            temp_sum = contrast_del_z_block_sum[tid];
        }
    }
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    if (tid == 0)
    {
        if (blockIdx.x == 0)
        {
            means[0] = temp_sum / num_elements;
        }
        else if (blockIdx.x == 1)
        {
            means[1] = temp_sum / num_elements;
        }
        else if (blockIdx.x == 2)
        {
            means[2] = temp_sum / num_elements;
        }
        else
        {
            means[3] = temp_sum / num_elements;
        }
    }
}

void getContrastDelBatchReduce(double *image,
                               double *image_contrast, double *image_del_theta_contrast,
                               int height, int width,
                               double *contrast_block_sum,
                               double *contrast_del_x_block_sum,
                               double *contrast_del_y_block_sum,
                               double *contrast_del_z_block_sum,
                               double *means,
                               int num_events,
                               cudaStream_t const *stream)
{
    // int dev = 0;
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    // int numBlocksPerSm = 0;
    // // Number of threads my_kernel will be launched with
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, getContrastDelBatchReduceHarder_, 512, 0);
    // std::cout << numBlocksPerSm<<deviceProp.multiProcessorCount << std::endl;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, getContrastDelBatchReduceHarder_, 256, 0);
    // std::cout << numBlocksPerSm<<deviceProp.multiProcessorCount << std::endl;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, getContrastDelBatchReduceHarder_, 128, 0);
    // std::cout << numBlocksPerSm<<deviceProp.multiProcessorCount << std::endl;

    // int blockSize = 512; // The launch configurator returned block size
    int blockSize = 256; // The launch configurator returned block size
    // int prev_gridsize = (num_events + blockSize - 1) / blockSize;
    int prev_gridsize = std::min(128, (num_events + blockSize - 1) / 512);

    // int gridSize = 85; // The actual grid size needed, based on input size
    // int gridSize = std::min(512, (height * width + blockSize - 1) / blockSize);
    int gridSize = 5 * 8;

    int smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(double) : blockSize * sizeof(double);

    // thrust::device_vector<double> imagedebug(height * width * 4);
    // double *imagedebugptr = thrust::raw_pointer_cast(imagedebug.data());
    // getContrastDelBatchReduceHarder_<<<gridSize, blockSize, smemSize>>>(image, height * width, means, contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum, prev_gridsize);
    int image_pixels = height * width;
    void *kernel_args[] = {
        (void *)&image,
        (void *)&image_pixels,
        (void *)&means,
        (void *)&contrast_block_sum,
        (void *)&contrast_del_x_block_sum,
        (void *)&contrast_del_y_block_sum,
        (void *)&contrast_del_z_block_sum,
        (void *)&prev_gridsize,
    };
    //    (void *)&imagedebugptr};
    // int dev = 0;
    // int supportsCoopLaunch = 0;
    // cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    // std::cout << supportsCoopLaunch << std::endl;

    checkCudaErrors(cudaPeekAtLastError());
    cudaLaunchCooperativeKernel((void *)getContrastDelBatchReduceHarder256_, gridSize, blockSize, kernel_args, smemSize);
    checkCudaErrors(cudaPeekAtLastError());

    // START OF DEBUG
    // std::cout << "harder1 sum " << -thrustSum(contrast_block_sum, gridSize) / (height * width)
    //           << " " << -2 * thrustSum(contrast_del_x_block_sum, gridSize) / (height * width)
    //           << " " << -2 * thrustSum(contrast_del_y_block_sum, gridSize) / (height * width)
    //           << " " << -2 * thrustSum(contrast_del_z_block_sum, gridSize) / (height * width)
    //           << std::endl;

    // END DEBUG
    checkCudaErrors(cudaPeekAtLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaDeviceSynchronize());
    // std::cout << "thrust hardreduce             " << -thrust::reduce(imagedebug.begin(), imagedebug.begin() + height * width, 0.0, thrust::plus<double>()) / (height * width) << " "
    //           << -2 * thrust::reduce(imagedebug.begin() + height * width, imagedebug.begin() + height * width * 2, 0.0, thrust::plus<double>()) / (height * width) << " "
    //           << -2 * thrust::reduce(imagedebug.begin() + height * width * 2, imagedebug.begin() + height * width * 3, 0.0, thrust::plus<double>()) / (height * width) << " "
    //           << -2 * thrust::reduce(imagedebug.begin() + height * width * 3, imagedebug.begin() + height * width * 4, 0.0, thrust::plus<double>()) / (height * width) << " "
    //           << std::endl;
    if (height == 180 && width == 240)
        getContrastDelBatchReduceHarderPt2_<8 * 5><<<4, 128, 128 * sizeof(double), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
    else if (height == 480 && width == 640)
        getContrastDelBatchReduceHarderPt2_<8 * 5><<<4, 512, 512 * sizeof(double), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
    else if (height == 400 && width == 400)
        getContrastDelBatchReduceHarderPt2_<8 * 5><<<4, 512, 512 * sizeof(double), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
    else if (height == 720 && width == 1280)
        getContrastDelBatchReduceHarderPt2_<8 * 5><<<4, 512, 512 * sizeof(double), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
    else
    {
        getContrastDelBatchReduceHarderPt2_<8 * 5><<<4, 512, 512 * sizeof(double), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
    }

    checkCudaErrors(cudaPeekAtLastError());
    cudaMemsetAsync(image, 0, (height) * (width) * sizeof(double) * 4, stream[1]);

    checkCudaErrors(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();
    {
        nvtx3::scoped_range r{"final contrast"};
        int num_el = height * width;
        image_contrast[0] = -contrast_block_sum[0] / num_el;
        image_del_theta_contrast[0] = -2 * contrast_block_sum[1] / num_el;
        image_del_theta_contrast[1] = -2 * contrast_block_sum[2] / num_el;
        image_del_theta_contrast[2] = -2 * contrast_block_sum[3] / num_el;
    }
}

__device__ double getRandom(uint64_t seed, int tid, int threadCallCount)
{
    curandState s;
    curand_init(seed + tid + threadCallCount, 0, 0, &s);
    // return curand_uniform(&s);
    return curand_log_normal(&s, 1e-16, 10.0);
}
__global__ void one_step_kernel_(uint64_t seed, double *randoms, int numel)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for (size_t idx = thread_grid_idx; idx < numel; idx += num_threads_in_grid)
    {

        randoms[idx] = getRandom(seed, idx, 0);
    }
}

void one_step_kernel(uint64_t seed, double *randoms, int numel)
{
    one_step_kernel_<<<43, 1024>>>(seed, randoms, numel);
}