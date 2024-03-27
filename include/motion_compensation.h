
#ifndef CUDACMAKE_TEST_H
#define CUDACMAKE_TEST_H
#include <stdint.h>
#include <reduce.h>
void fillImage(float fx,float fy,float cx,float cy, int height,int width, int num_events, float*x_unprojected,float*y_unprojected,float*x_prime,float*y_prime,float*t,float*image,const float rotation_x,const float rotation_y,const float rotation_z,bool do_jacobian, float* image_del_x, float* image_del_y, float* image_del_z);
void fillImageBilinear(float fx,float fy,float cx,float cy, int height,int width, int num_events, float*x_unprojected,float*y_unprojected,float*x_prime,float*y_prime,float*t,float*image,const float rotation_x,const float rotation_y,const float rotation_z,bool do_jacobian, float* image_del_x, float* image_del_y, float* image_del_z);
void fillImageBilinearIntrinsics(float fx,float fy,float cx,float cy, int height,int width, int num_events, float*x_unprojected,float*y_unprojected,float*x_prime,float*y_prime,float*t,float*image,const float rotation_x,const float rotation_y,const float rotation_z,bool do_jacobian, float* image_del_x, float* image_del_y, float* image_del_z);
float getMean(float *image,int height,int width, int cub_temp_size=0);
void subtractMean(float *image,int height,int width, int cub_temp_size=0);
float getContrast(float *image,int height,int width, int cub_temp_size=0);
float getContrastDel(float *image,float *image_del,int height,int width, int cub_temp_size=0);
void fillImageKronecker(int height, int width, int num_events,float *x_prime, float *y_prime,float *image);
int getCubSize(float *image, int height, int width);
float getMax(float *image, int height, int width);
void fillImageBilinearSeparate(float fx,float fy,float cx,float cy, int height,int width, int num_events, float*x_unprojected,float*y_unprojected,float*x_prime,float*y_prime,float*t,float*image,const float rotation_x,const float rotation_y,const float rotation_z,bool do_jacobian, float* image_del_x, float* image_del_y, float* image_del_z);
void getContrastDelBatch(float *image, float *image_del_theta_x,float *image_del_theta_y,float *image_del_theta_z, 
double *image_contrast, double *image_del_theta_contrast,
int height, int width, int cub_temp_size);
#endif //CUDACMAKE_TEST_H