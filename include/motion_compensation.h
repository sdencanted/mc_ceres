
#ifndef CUDACMAKE_TEST_H
#define CUDACMAKE_TEST_H
#include <stdint.h>
#include <reduce.h>
#include <nvtx3/nvtx3.hpp>
void fillImageBilinear(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z,  float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum);


void fillImageKronecker(int height, int width, int num_events, float *x_prime, float *y_prime, float *image);
float getMax(float *image, int height, int width);                    
void getContrastDelBatchReduce(float *image, 
                               double *image_contrast, double *image_del_theta_contrast,
                               int height, int width,
                               float *contrast_block_sum,
                               float *contrast_del_x_block_sum,
                               float *contrast_del_y_block_sum,
                               float *contrast_del_z_block_sum,
                               float *means,
                               float *contrast_block_sum_cpu,
                               int num_events);
void one_step_kernel(uint64_t seed, float* randoms, int numel);

#endif // CUDACMAKE_TEST_H