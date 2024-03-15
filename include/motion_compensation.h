
#ifndef CUDACMAKE_TEST_H
#define CUDACMAKE_TEST_H
#include <stdint.h>
#include <reduce.h>
void fillImage(double fx,double fy,double cx,double cy, int height,int width, int num_events, double*x_unprojected,double*y_unprojected,double*x_prime,double*y_prime,double*t,double*image,const double rotation_x,const double rotation_y,const double rotation_z,bool do_jacobian, double* image_del_x, double* image_del_y, double* image_del_z);
double getMean(double *image,int height,int width, int cub_temp_size=0);
void subtractMean(double *image,int height,int width, int cub_temp_size=0);
double getContrast(double *image,int height,int width, int cub_temp_size=0);
double getContrastDel(double *image,double *image_del,int height,int width, int cub_temp_size=0);
void fillImageKronecker(int height, int width, int num_events,double *x_prime, double *y_prime,double *image);
int getCubSize(double *image, int height, int width);
#endif //CUDACMAKE_TEST_H