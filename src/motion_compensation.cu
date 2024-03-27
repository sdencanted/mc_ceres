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

__global__ void fillImage_(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        float gaussian;
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

        float del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
        if (do_jacobian)
        {
            float fx_div_z_rotated = fx * z_rotated_inv;
            float fy_div_z_rotated = fy * z_rotated_inv;
            del_x_del_theta_y = fx_div_z_rotated * t[i] * (1 + x_unprojected[i] * x_rotated_norm);
            del_x_del_theta_z = fx_div_z_rotated * (-t[i] * y_unprojected[i]);
            del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
            del_y_del_theta_x = fy_div_z_rotated * t[i] * (-1 - y_unprojected[i] * y_rotated_norm);
            del_y_del_theta_z = fy_div_z_rotated * (t[i] * x_unprojected[i]);
            del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;
        }
        // populate image

        // Pseudo Gaussian
        // check if coordinates are 3 pixels in of the boundary
        int x_round = round(x_prime[i]);
        int y_round = round(y_prime[i]);
        if (x_round >= 1 && x_round <= width && y_round >= 1 && y_round <= height)
        {
            for (int row = y_round - 3; row < y_round + 4; row++)
            {
                for (int col = x_round - 3; col < x_round + 4; col++)
                {
                    // TODO: make a LUT for the values here rounded to a certain s.f. and see if there is a speed-up
                    float x_diff = col - x_prime[i];
                    float y_diff = row - y_prime[i];
                    // float x_diff = col - x_unprojected[i];
                    // float y_diff = row - y_unprojected[i];
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

void fillImage(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    fillImage_<<<blocks_per_sm * num_sm, threads_per_block>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, do_jacobian, image_del_x, image_del_y, image_del_z);
}
__global__ void fillImageBilinearIntrinsics_(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        // calculate theta x,y,z
        float theta_x_t = __fmul_rn(rotation_x, t[i]);
        float theta_y_t = __fmul_rn(rotation_y, t[i]);
        float theta_z_t = __fmul_rn(rotation_z, t[i]);

        // calculate x/y/z_rotated
        float z_rotated_inv = __fdiv_rn(1, (__fmul_rn(-theta_y_t, x_unprojected[i]) + __fmaf_rn(theta_x_t, y_unprojected[i], 1)));
        float x_rotated_norm = __fmul_rn(__fmaf_rn(-theta_z_t, y_unprojected[i], __fadd_rn(x_unprojected[i], theta_y_t)), z_rotated_inv);
        float y_rotated_norm = __fmul_rn(__fmaf_rn(theta_z_t, x_unprojected[i], __fsub_rn(y_unprojected[i], theta_x_t)), z_rotated_inv);

        // calculate x_prime and y_prime
        x_prime[i] = __fmaf_rn(fx, x_rotated_norm, cx);
        y_prime[i] = __fmaf_rn(fy, y_rotated_norm, cy);

        float del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
        // populate image

        // Bilinear
        // check if coordinates are 3 pixels in of the boundary
        int x_trunc = int(x_prime[i]);
        int y_trunc = int(y_prime[i]);
        if (x_trunc >= 1 && x_trunc <= width - 2 && y_trunc >= 1 && y_trunc <= height - 2)
        {
            int idx1 = __fmaf_rn(__fsub_rn(y_trunc, 1), width, __fsub_rn(x_trunc, 1));
            int idx2 = __fmaf_rn(__fsub_rn(y_trunc, 1), width, x_trunc);
            int idx3 = __fmaf_rn((y_trunc), width, __fsub_rn(x_trunc, 1));
            int idx4 = __fmaf_rn((y_trunc), width, x_trunc);
            float x_diff = __fsub_rn(x_prime[i], x_trunc);
            float y_diff = __fsub_rn(y_prime[i], y_trunc);
            atomicAdd(&image[idx1], __fmul_rn(__fsub_rn(1, x_diff), __fsub_rn(1, y_diff)));
            atomicAdd(&image[idx2], __fmul_rn((x_diff), __fsub_rn(1, y_diff)));
            atomicAdd(&image[idx3], __fsub_rn(1, x_diff) * (y_diff));
            atomicAdd(&image[idx4], __fmul_rn((x_diff), (y_diff)));

            if (do_jacobian)
            {
                float fx_div_z_rotated = __fmul_rn(fx, z_rotated_inv);
                float fy_div_z_rotated = __fmul_rn(fy, z_rotated_inv);
                del_x_del_theta_y = __fmul_rn(__fmul_rn(fx_div_z_rotated, t[i]), __fmaf_rn(x_unprojected[i], x_rotated_norm, 1));
                del_x_del_theta_z = __fmul_rn(fx_div_z_rotated, __fmul_rn(-t[i], y_unprojected[i]));
                del_x_del_theta_x = __fmul_rn(del_x_del_theta_z, x_rotated_norm);
                del_y_del_theta_x = __fmul_rn(__fmul_rn(fy_div_z_rotated, t[i]), __fmaf_rn(-y_unprojected[i], y_rotated_norm, -1));
                del_y_del_theta_z = __fmul_rn(fy_div_z_rotated, __fmul_rn(t[i], x_unprojected[i]));
                del_y_del_theta_y = __fmul_rn(del_y_del_theta_z, y_rotated_norm);
                float d1x = __fsub_rn(y_diff, 1);
                float d1y = __fsub_rn(x_diff, 1);
                float d2x = __fsub_rn(1, y_diff);
                float d2y = -x_diff;
                float d3x = -y_diff;
                float d3y = __fsub_rn(1, x_diff);
                float d4x = y_diff;
                float d4y = x_diff;
                atomicAdd(&image_del_x[idx1], __fmaf_rn(d1x, del_x_del_theta_x, __fmul_rn(d1y, del_y_del_theta_x)));
                atomicAdd(&image_del_x[idx2], __fmaf_rn(d2x, del_x_del_theta_x, __fmul_rn(d2y, del_y_del_theta_x)));
                atomicAdd(&image_del_x[idx3], __fmaf_rn(d3x, del_x_del_theta_x, __fmul_rn(d3y, del_y_del_theta_x)));
                atomicAdd(&image_del_x[idx4], __fmaf_rn(d4x, del_x_del_theta_x, __fmul_rn(d4y, del_y_del_theta_x)));

                atomicAdd(&image_del_y[idx1], __fmaf_rn(d1x, del_x_del_theta_y, __fmul_rn(d1y, del_y_del_theta_y)));
                atomicAdd(&image_del_y[idx2], __fmaf_rn(d2x, del_x_del_theta_y, __fmul_rn(d2y, del_y_del_theta_y)));
                atomicAdd(&image_del_y[idx3], __fmaf_rn(d3x, del_x_del_theta_y, __fmul_rn(d3y, del_y_del_theta_y)));
                atomicAdd(&image_del_y[idx4], __fmaf_rn(d4x, del_x_del_theta_y, __fmul_rn(d4y, del_y_del_theta_y)));

                atomicAdd(&image_del_z[idx1], __fmaf_rn(d1x, del_x_del_theta_z, __fmul_rn(d1y, del_y_del_theta_z)));
                atomicAdd(&image_del_z[idx2], __fmaf_rn(d2x, del_x_del_theta_z, __fmul_rn(d2y, del_y_del_theta_z)));
                atomicAdd(&image_del_z[idx3], __fmaf_rn(d3x, del_x_del_theta_z, __fmul_rn(d3y, del_y_del_theta_z)));
                atomicAdd(&image_del_z[idx4], __fmaf_rn(d4x, del_x_del_theta_z, __fmul_rn(d4y, del_y_del_theta_z)));
            }
        }
    }
}
void fillImageBilinearIntrinsics(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{
    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       fillImageBilinearIntrinsics_, 0, 0);
    // Round up according to array size
    gridSize = (num_events + blockSize - 1) / blockSize;
    fillImageBilinearIntrinsics_<<<gridSize, blockSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, do_jacobian, image_del_x, image_del_y, image_del_z);
}

__global__ void fillImageBilinear_(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
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
        // check if coordinates are 3 pixels in of the boundary
        int x_trunc = int(x_prime[i]);
        int y_trunc = int(y_prime[i]);
        if (x_trunc >= 1 && x_trunc <= width - 2 && y_trunc >= 1 && y_trunc <= height - 2)
        {

            int idx1 = x_trunc - 1 + (y_trunc - 1) * width;
            int idx2 = x_trunc + (y_trunc - 1) * width;
            int idx3 = x_trunc - 1 + (y_trunc)*width;
            int idx4 = x_trunc + (y_trunc)*width;
            float x_diff = x_prime[i] - x_trunc;
            float y_diff = y_prime[i] - y_trunc;
            atomicAdd(&image[idx1], (1 - x_diff) * (1 - y_diff));
            atomicAdd(&image[idx2], (x_diff) * (1 - y_diff));
            atomicAdd(&image[idx3], (1 - x_diff) * (y_diff));
            atomicAdd(&image[idx4], (x_diff) * (y_diff));
            if (do_jacobian)
            {

                float del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
                float fx_div_z_rotated = fx * z_rotated_inv;
                float fy_div_z_rotated = fy * z_rotated_inv;
                del_x_del_theta_y = fx_div_z_rotated * t[i] * (1 + x_unprojected[i] * x_rotated_norm);
                del_x_del_theta_z = fx_div_z_rotated * (-t[i] * y_unprojected[i]);
                del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
                del_y_del_theta_x = fy_div_z_rotated * t[i] * (-1 - y_unprojected[i] * y_rotated_norm);
                del_y_del_theta_z = fy_div_z_rotated * (t[i] * x_unprojected[i]);
                del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;
                float d1x = -(1 - y_diff);
                float d1y = -(1 - x_diff);
                float d2x = 1 - y_diff;
                float d2y = -x_diff;
                float d3x = -y_diff;
                float d3y = 1 - x_diff;
                float d4x = y_diff;
                float d4y = x_diff;
                atomicAdd(&image_del_x[idx1], d1x * del_x_del_theta_x + d1y * del_y_del_theta_x);
                atomicAdd(&image_del_x[idx2], d2x * del_x_del_theta_x + d2y * del_y_del_theta_x);
                atomicAdd(&image_del_x[idx3], d3x * del_x_del_theta_x + d3y * del_y_del_theta_x);
                atomicAdd(&image_del_x[idx4], d4x * del_x_del_theta_x + d4y * del_y_del_theta_x);

                atomicAdd(&image_del_y[idx1], d1x * del_x_del_theta_y + d1y * del_y_del_theta_y);
                atomicAdd(&image_del_y[idx2], d2x * del_x_del_theta_y + d2y * del_y_del_theta_y);
                atomicAdd(&image_del_y[idx3], d3x * del_x_del_theta_y + d3y * del_y_del_theta_y);
                atomicAdd(&image_del_y[idx4], d4x * del_x_del_theta_y + d4y * del_y_del_theta_y);

                atomicAdd(&image_del_z[idx1], d1x * del_x_del_theta_z + d1y * del_y_del_theta_z);
                atomicAdd(&image_del_z[idx2], d2x * del_x_del_theta_z + d2y * del_y_del_theta_z);
                atomicAdd(&image_del_z[idx3], d3x * del_x_del_theta_z + d3y * del_y_del_theta_z);
                atomicAdd(&image_del_z[idx4], d4x * del_x_del_theta_z + d4y * del_y_del_theta_z);
            }
        }
    }
}
void fillImageBilinear(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{
    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       fillImageBilinear_, 0, 0);
    // Round up according to array size
    gridSize = (num_events + blockSize - 1) / blockSize;
    fillImageBilinear_<<<gridSize, blockSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, do_jacobian, image_del_x, image_del_y, image_del_z);
}

__global__ void warpBilinear_(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z, float *z_rotated_inv, float *x_rotated_norm, float *y_rotated_norm)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        // calculate theta x,y,z
        float theta_x_t = rotation_x * t[i];
        float theta_y_t = rotation_y * t[i];
        float theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        z_rotated_inv[i] = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        x_rotated_norm[i] = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv[i];
        y_rotated_norm[i] = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv[i];

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm[i] + cx;
        y_prime[i] = fy * y_rotated_norm[i] + cy;
    }
}

__global__ void fillImageOnly_(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z, float *z_rotated_inv, float *x_rotated_norm, float *y_rotated_norm)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        // Bilinear
        // check if coordinates are 3 pixels in of the boundary
        int x_trunc = int(x_prime[i]);
        int y_trunc = int(y_prime[i]);
        if (x_trunc >= 1 && x_trunc <= width - 2 && y_trunc >= 1 && y_trunc <= height - 2)
        {
            int idx1 = x_trunc - 1 + (y_trunc - 1) * width;
            int idx2 = x_trunc + (y_trunc - 1) * width;
            int idx3 = x_trunc - 1 + (y_trunc)*width;
            int idx4 = x_trunc + (y_trunc)*width;
            float x_diff = x_prime[i] - x_trunc;
            float y_diff = y_prime[i] - y_trunc;
            atomicAdd(&image[idx1], (1 - x_diff) * (1 - y_diff));
            atomicAdd(&image[idx2], (x_diff) * (1 - y_diff));
            atomicAdd(&image[idx3], (1 - x_diff) * (y_diff));
            atomicAdd(&image[idx4], (x_diff) * (y_diff));

            if (do_jacobian)
            {

                float del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
                float fx_div_z_rotated = fx * z_rotated_inv[i];
                float fy_div_z_rotated = fy * z_rotated_inv[i];
                del_x_del_theta_y = fx_div_z_rotated * t[i] * (1 + x_unprojected[i] * x_rotated_norm[i]);
                del_x_del_theta_z = fx_div_z_rotated * (-t[i] * y_unprojected[i]);
                del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm[i];
                del_y_del_theta_x = fy_div_z_rotated * t[i] * (-1 - y_unprojected[i] * y_rotated_norm[i]);
                del_y_del_theta_z = fy_div_z_rotated * (t[i] * x_unprojected[i]);
                del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm[i];
                float d1x = -(1 - y_diff);
                float d1y = -(1 - x_diff);
                float d2x = 1 - y_diff;
                float d2y = -x_diff;
                float d3x = -y_diff;
                float d3y = 1 - x_diff;
                float d4x = y_diff;
                float d4y = x_diff;
                atomicAdd(&image_del_x[idx1], d1x * del_x_del_theta_x + d1y * del_y_del_theta_x);
                atomicAdd(&image_del_x[idx2], d2x * del_x_del_theta_x + d2y * del_y_del_theta_x);
                atomicAdd(&image_del_x[idx3], d3x * del_x_del_theta_x + d3y * del_y_del_theta_x);
                atomicAdd(&image_del_x[idx4], d4x * del_x_del_theta_x + d4y * del_y_del_theta_x);

                atomicAdd(&image_del_y[idx1], d1x * del_x_del_theta_y + d1y * del_y_del_theta_y);
                atomicAdd(&image_del_y[idx2], d2x * del_x_del_theta_y + d2y * del_y_del_theta_y);
                atomicAdd(&image_del_y[idx3], d3x * del_x_del_theta_y + d3y * del_y_del_theta_y);
                atomicAdd(&image_del_y[idx4], d4x * del_x_del_theta_y + d4y * del_y_del_theta_y);

                atomicAdd(&image_del_z[idx1], d1x * del_x_del_theta_z + d1y * del_y_del_theta_z);
                atomicAdd(&image_del_z[idx2], d2x * del_x_del_theta_z + d2y * del_y_del_theta_z);
                atomicAdd(&image_del_z[idx3], d3x * del_x_del_theta_z + d3y * del_y_del_theta_z);
                atomicAdd(&image_del_z[idx4], d4x * del_x_del_theta_z + d4y * del_y_del_theta_z);
            }
        }
    }
}
void fillImageBilinearSeparate(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, bool do_jacobian, float *image_del_x, float *image_del_y, float *image_del_z)
{

    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       warpBilinear_, 0, 0);
    // Round up according to array size
    gridSize = (num_events + blockSize - 1) / blockSize;
    float *z_rotated_inv;
    float *x_rotated_norm;
    float *y_rotated_norm;

    cudaMalloc(&z_rotated_inv, sizeof(float) * num_events);
    cudaMalloc(&x_rotated_norm, sizeof(float) * num_events);
    cudaMalloc(&y_rotated_norm, sizeof(float) * num_events);
    warpBilinear_<<<gridSize, blockSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, do_jacobian, image_del_x, image_del_y, image_del_z, z_rotated_inv, x_rotated_norm, y_rotated_norm);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       fillImageOnly_, 0, 0);
    // Round up according to array size
    gridSize = (num_events + blockSize - 1) / blockSize;
    fillImageOnly_<<<gridSize, blockSize>>>(fx, fy, cx, cy, height, width, num_events, x_unprojected, y_unprojected, x_prime, y_prime, t, image, rotation_x, rotation_y, rotation_z, do_jacobian, image_del_x, image_del_y, image_del_z, z_rotated_inv, x_rotated_norm, y_rotated_norm);
    checkCudaErrors(cudaFree(z_rotated_inv));
    checkCudaErrors(cudaFree(x_rotated_norm));
    checkCudaErrors(cudaFree(y_rotated_norm));
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
int getCubSize(float *image, int height, int width)
{
    size_t temp_cub_temp_size;
    float *temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, image, (height) * (width), cub::Sum(), 0);
    return temp_cub_temp_size;
}
float getMean(float *image, int height, int width, int cub_temp_size)
{

    float *out;
    cudaMalloc(&out, sizeof(float));
    size_t temp_cub_temp_size;
    float *temp_storage = NULL;

    if (cub_temp_size != 0)
    {
        temp_cub_temp_size = cub_temp_size;
    }
    else
    {
        std::cout << "no cub temp size" << std::endl;
        cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height) * (width), cub::Sum(), 0);
        cudaDeviceSynchronize();
    }
    checkCudaErrors(cudaMalloc(&temp_storage, temp_cub_temp_size));
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, image, out, (height) * (width), cub::Sum(), 0);
    cudaDeviceSynchronize();
    float sum;
    checkCudaErrors(cudaMemcpy(&sum, out, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(temp_storage));

    return sum / ((height) * (width));
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
float getMeanCpu(float *image, int height, int width)
{

    float sum = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
            sum += image[i * (width) + j];
    }
    return sum / ((height) * (width));
}
float getMeanCpuCrop(float *image, int height, int width)
{

    float sum = 0;
    for (int i = 3; i < height + 3; i++)
    {
        for (int j = 3; j < width + 3; j++)
        {
            sum += image[i * (width + 6) + j];
        }
    }
    return sum / (height * width);
}
__global__ void subtractMean_(float *image, int num_elements, float mean)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for (size_t i = thread_grid_idx; i < num_elements; i += num_threads_in_grid)
    {
        image[i] -= mean;
        // atomicAdd(&image[i], -mean);
    }
}
void subtractMean(float *image, int height, int width, int cub_temp_size)
{

    // cudaEvent_t start, stop;
    // float time_ms;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    float mean_cuda = getMean(image, height, width, cub_temp_size);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_ms, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       subtractMean_, 0, 0);
    // Round up according to array size
    gridSize = ((height) * (width) + blockSize - 1) / blockSize;

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    subtractMean_<<<gridSize, blockSize>>>(image, height * width, mean_cuda);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time_ms, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // std::cout<<"actual subtractmean time: "<<time_ms<<std::endl;
}
__global__ void getContrast_(float *image, int num_elements, float *image_out)
{
    // size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    // size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // for (size_t i = thread_grid_idx; i < num_elements; i += num_threads_in_grid)
    // {
    //     image_out[i] = image[i] * image[i];
    // }


    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    
    image_out[idx] = image[idx] * image[idx];
    // if (idx < num_elements)
    // {
    //     image_out[idx] = image[idx] * image[idx];
    // }
}
float getContrast(float *image, int height, int width, int cub_temp_size)
{
    float contrast = 0;

    float *temp_image;
    checkCudaErrors(cudaMalloc((void **)&temp_image, (unsigned int)sizeof(float) * (height) * (width)));
    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize=57;   // The launch configurator returned block size
    // int minGridSize; // The minimum grid size needed to achieve the
    //                  // maximum occupancy for a full device launch
    int gridSize=768;    // The actual grid size needed, based on input size

    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
    //                                    getContrast_, 0, 0);
    // Round up according to array size
    // gridSize = (height * width + blockSize - 1) / blockSize;
    getContrast_<<<gridSize, blockSize>>>(image, (height) * (width), temp_image);
    // std::cout<<"getContrast "<< gridSize<<" "<< blockSize<<std::endl;
    contrast = getMean(temp_image, height, width, cub_temp_size);
    checkCudaErrors(cudaFree(temp_image));
    return contrast;
}

__global__ void getContrastDel_(float *image, float *image_del, int num_elements)
{
    // size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    // size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // for (size_t i = thread_grid_idx; i <  num_elements; i += num_threads_in_grid)
    // {
    //     image_del[i] *= image[i];
    // }
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    image_del[idx] *= image[idx];
    // if (idx < num_elements)
    // {
    //     image_del[idx] *= image[idx];
    // }
}
float getContrastDel(float *image, float *image_del, int height, int width, int cub_temp_size)
{
    float contrast = 0;
    // const int num_sm = 8; // Jetson Orin NX
    // const int blocks_per_sm = 4;
    // const int threads_per_block = 128;
    int blockSize=57;   // The launch configurator returned block size
    // int minGridSize; // The minimum grid size needed to achieve the
    //                  // maximum occupancy for a full device launch
    int gridSize=768;    // The actual grid size needed, based on input size

    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
    //                                    getContrastDel_, 0, 0);
    // Round up according to array size
    // gridSize = (height * width + blockSize - 1) / blockSize;
    // std::cout<<"getContrastDel "<< gridSize<<" "<< blockSize<<std::endl;
    getContrastDel_<<<gridSize, blockSize>>>(image, image_del, height* width);
    contrast = 2 * getMean(image_del, height, width, cub_temp_size);
    return contrast;
}