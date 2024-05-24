
#ifndef MOTION_COMPENSATION_DOUBLE_H
#define MOTION_COMPENSATION_DOUBLE_H
#include <stdint.h>
#include <reduce.h>
#include <nvtx3/nvtx3.hpp>
void fillImage(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z,  double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum);

void fillImageBilinear(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, double *image, const double rotation_x, const double rotation_y, const double rotation_z,  double *contrast_block_sum, double *contrast_del_x_block_sum, double *contrast_del_y_block_sum, double *contrast_del_z_block_sum);


void fillImageKronecker(int height, int width, int num_events, double *x_prime, double *y_prime, double *image);
double getMax(double *image, int height, int width);                    
void getContrastDelBatchReduce(double *image, 
                               double *image_contrast, double *image_del_theta_contrast,
                               int height, int width,
                               double *contrast_block_sum,
                               double *contrast_del_x_block_sum,
                               double *contrast_del_y_block_sum,
                               double *contrast_del_z_block_sum,
                               double *means,
                               int num_events,
                               cudaStream_t const* stream);
void one_step_kernel(uint64_t seed, double* randoms, int numel);
double thrustMean(double* image_,int height_,int width_);
void warpEvents(double fx, double fy, double cx, double cy, int height, int width, int num_events, double *x_unprojected, double *y_unprojected, double *x_prime, double *y_prime, double *t, const double rotation_x, const double rotation_y, const double rotation_z);

#endif // MOTION_COMPENSATION_H