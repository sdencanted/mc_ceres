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
#include <curand.h>
#include <curand_kernel.h>
#define FULL_MASK 0xffffffff
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

__global__ void fillImageBilinear_(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum)
{

    float image_sum = 0;
    float image_sum_del_theta_x = 0;
    float image_sum_del_theta_y = 0;
    float image_sum_del_theta_z = 0;
    float *image_del_x = image + height * width;
    float *image_del_y = image + height * width * 2;
    float *image_del_z = image + height * width * 3;
    size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    // size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    if (i < num_events)
    {
        // calculate theta x,y,z
        float theta_x_t = rotation_x * t[i];
        float theta_y_t = rotation_y * t[i];
        float theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        float z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        float x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        float y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

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
            float x_diff = x_prime[i] - x_trunc;
            float y_diff = y_prime[i] - y_trunc;
            float del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
            float fx_div_z_rotated_ti = fx * z_rotated_inv * t[i];
            float fy_div_z_rotated_ti = fy * z_rotated_inv * t[i];
            del_x_del_theta_y = fx_div_z_rotated_ti * (1 + x_unprojected[i] * x_rotated_norm);
            del_x_del_theta_z = fx_div_z_rotated_ti * -y_unprojected[i];
            del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
            del_y_del_theta_x = fy_div_z_rotated_ti * (-1 - y_unprojected[i] * y_rotated_norm);
            del_y_del_theta_z = fy_div_z_rotated_ti * x_unprojected[i];
            del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;
            // float d1x = -(1 - y_diff);
            // float d1y = -(1 - x_diff);
            float d2x = 1 - y_diff;
            float d2y = -x_diff;
            float d3x = -y_diff;
            float d3y = 1 - x_diff;
            float d4x = y_diff;
            float d4y = x_diff;

            float d1x = -d2x;
            float d1y = -d3y;

            // float im1 = (1 - x_diff) * (1 - y_diff);
            float im1 = d3y * d2x;
            // float im2 = (x_diff) * (1 - y_diff);
            float im2 = d4y * d2x;
            // float im3 = (1 - x_diff) * (y_diff);
            float im3 = d3y * y_diff;
            float im4 = (x_diff) * (y_diff);
            image_sum = im1 + im2 + im3 + im4;
            atomicAdd(&image[idx1], im1);
            atomicAdd(&image[idx2], im2);
            atomicAdd(&image[idx3], im3);
            atomicAdd(&image[idx4], im4);
            float dx1 = d1x * del_x_del_theta_x + d1y * del_y_del_theta_x;
            float dx2 = d2x * del_x_del_theta_x + d2y * del_y_del_theta_x;
            float dx3 = d3x * del_x_del_theta_x + d3y * del_y_del_theta_x;
            float dx4 = d4x * del_x_del_theta_x + d4y * del_y_del_theta_x;
            image_sum_del_theta_x = dx1 + dx2 + dx3 + dx4;

            atomicAdd(&image_del_x[idx1], dx1);
            atomicAdd(&image_del_x[idx2], dx2);
            atomicAdd(&image_del_x[idx3], dx3);
            atomicAdd(&image_del_x[idx4], dx4);
            float dy1 = d1x * del_x_del_theta_y + d1y * del_y_del_theta_y;
            float dy2 = d2x * del_x_del_theta_y + d2y * del_y_del_theta_y;
            float dy3 = d3x * del_x_del_theta_y + d3y * del_y_del_theta_y;
            float dy4 = d4x * del_x_del_theta_y + d4y * del_y_del_theta_y;
            image_sum_del_theta_y = dy1 + dy2 + dy3 + dy4;
            atomicAdd(&image_del_y[idx1], dy1);
            atomicAdd(&image_del_y[idx2], dy2);
            atomicAdd(&image_del_y[idx3], dy3);
            atomicAdd(&image_del_y[idx4], dy4);
            float dz1 = d1x * del_x_del_theta_z + d1y * del_y_del_theta_z;
            float dz2 = d2x * del_x_del_theta_z + d2y * del_y_del_theta_z;
            float dz3 = d3x * del_x_del_theta_z + d3y * del_y_del_theta_z;
            float dz4 = d4x * del_x_del_theta_z + d4y * del_y_del_theta_z;
            image_sum_del_theta_z = dz1 + dz2 + dz3 + dz4;
            atomicAdd(&image_del_z[idx1], dz1);
            atomicAdd(&image_del_z[idx2], dz2);
            atomicAdd(&image_del_z[idx3], dz3);
            atomicAdd(&image_del_z[idx4], dz4);
        }
    }
    float *sdata = SharedMemory<float>();
    uint16_t tid = threadIdx.x;

    // do reduction in shared mem

    // sum up to 128 elements

    float temp_sum;
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
void fillImageBilinear(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum)
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
    gridSize = (num_events + blockSize - 1) / blockSize;

    int smemSize = blockSize * sizeof(float);
    fillImageBilinear_<<<gridSize, blockSize, smemSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
}

__global__ void fillImageKronecker_(int height, int width, int num_events, float *x_prime, float *y_prime, float *image)
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
void fillImageKronecker(int height, int width, int num_events, float *x_prime, float *y_prime, float *image)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    cudaMemset(image, 0, height * width * sizeof(float));
    fillImageKronecker_<<<blocks_per_sm * num_sm, threads_per_block>>>(height, width, num_events, x_prime, y_prime, image);
}
float getMax(float *image, int height, int width)
{
    float *out;
    cudaMalloc(&out, sizeof(float));
    size_t temp_cub_temp_size;
    float *temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height) * (width), cub::Max(), 0);
    cudaDeviceSynchronize();
    cudaMalloc(&temp_storage, temp_cub_temp_size);
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height) * (width), cub::Max(), 0);
    cudaDeviceSynchronize();
    float maximum;
    cudaMemcpy(&maximum, out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(out);
    cudaFree(temp_storage);
    return maximum;
}

__global__ void getContrastDelBatchReduceHarder_(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize)
{

    float *image_del_x = image + num_elements;
    float *image_del_y = image + num_elements * 2;
    float *image_del_z = image + num_elements * 3;
    // START COPY
    float *sdata = SharedMemory<float>();
    float temp_sum = 0;
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

    // END COPY
    float image_contrast = 0;
    float image_contrast_del_theta_x = 0;
    float image_contrast_del_theta_y = 0;
    float image_contrast_del_theta_z = 0;
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    // size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    size_t idx = thread_grid_idx;
    __syncthreads();
    while (idx < num_elements)
    {
        float image_norm = image[idx] - means[0];
        float image_norm_x = image_del_x[idx] - means[1];
        float image_norm_y = image_del_y[idx] - means[2];
        float image_norm_z = image_del_z[idx] - means[3];
        image_contrast = image_norm * image_norm;
        image_contrast_del_theta_x = image_norm * image_norm_x;
        image_contrast_del_theta_y = image_norm * image_norm_y;
        image_contrast_del_theta_z = image_norm * image_norm_z;
        idx += blockDim.x * gridDim.x;
    }
    // float *sdata = SharedMemory<float>();
    // uint16_t tid = threadIdx.x;

    // do reduction in shared mem

    // sum up to 128 elements

    // float temp_sum;
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
        image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 128];
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
__global__ void getContrastDelBatchReduceHarderPt2_(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum)
{
    float *sdata = SharedMemory<float>();
    float temp_sum;
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
        else
        {
            temp_sum = contrast_del_z_block_sum[tid];
        }
    }
    else
    {
        temp_sum = 0;
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
__global__ void meanPt2_(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int num_elements, float *means, int prev_gridsize)
{
    float *sdata = SharedMemory<float>();
    float temp_sum = 0;
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

void getContrastDelBatchReduce(float *image,
                               double *image_contrast, double *image_del_theta_contrast,
                               int height, int width,
                               float *contrast_block_sum,
                               float *contrast_del_x_block_sum,
                               float *contrast_del_y_block_sum,
                               float *contrast_del_z_block_sum,
                               float *means,
                               int num_events,
                               cudaStream_t const *stream)
{
    int blockSize = 512; // The launch configurator returned block size
    int prev_gridsize = (num_events + blockSize - 1) / blockSize;
    // int gridSize = 85; // The actual grid size needed, based on input size
    int gridSize = std::min(512, (height * width + blockSize - 1) / blockSize);

    int smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(float) : blockSize * sizeof(float);

    // meanPt2_<<<4, 128, 128 * sizeof(float)>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum, height * width, means,prev_gridsize);

    getContrastDelBatchReduceHarder_<<<gridSize, blockSize, smemSize>>>(image, height * width, means, contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum, prev_gridsize);
    if (height == 180 && width == 240)
        getContrastDelBatchReduceHarderPt2_<85><<<4, 512, 512 * sizeof(float), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);
    else if (height == 480 && width == 640)
        getContrastDelBatchReduceHarderPt2_<512><<<4, 512, 512 * sizeof(float), stream[0]>>>(contrast_block_sum, contrast_del_x_block_sum, contrast_del_y_block_sum, contrast_del_z_block_sum);

    cudaMemsetAsync(image, 0, (height) * (width) * sizeof(float) * 4, stream[1]);
    cudaDeviceSynchronize();
    {

        nvtx3::scoped_range r{"final contrast"};
        int num_el = height * width;
        image_contrast[0] = -contrast_block_sum[0] / num_el;
        image_del_theta_contrast[0] = -2 * contrast_block_sum[1] / num_el;
        image_del_theta_contrast[1] = -2 * contrast_block_sum[2] / num_el;
        image_del_theta_contrast[2] = -2 * contrast_block_sum[3] / num_el;
    }
}

__device__ float getRandom(uint64_t seed, int tid, int threadCallCount)
{
    curandState s;
    curand_init(seed + tid + threadCallCount, 0, 0, &s);
    // return curand_uniform(&s);
    return curand_log_normal(&s, 1e-16, 10.0);
}
__global__ void one_step_kernel_(uint64_t seed, float *randoms, int numel)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for (size_t idx = thread_grid_idx; idx < numel; idx += num_threads_in_grid)
    {

        randoms[idx] = getRandom(seed, idx, 0);
    }
}

void one_step_kernel(uint64_t seed, float *randoms, int numel)
{
    one_step_kernel_<<<43, 1024>>>(seed, randoms, numel);
}