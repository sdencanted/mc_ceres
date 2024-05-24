#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
#include "glog/logging.h"

// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_double.h"
#include "mc_gradient_double.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{

    std::cout.precision(std::numeric_limits<double>::digits10 + 1);
    // int height = 180;
    // int width = 240;
    int height = 720;
    int width = 1280;
    double lower_bound = -5 * 2 * M_PI;
    double upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    // double fx = 199.092366542, fy = 198.82882047, cx = 132.192071378, cy = 110.712660011; // boxes
    double fx = 3.22418800e+03, fy = 3.21510040e+03, cx = (8.80357033e+02), cy = (4.17066114e+02) ; // evk4
    google::InitGoogleLogging(argv[0]);

    // load csv to x,y,t
    // std::ifstream events_str("boxes_rotation.csv", std::ifstream::in);
    std::ifstream events_str("bag_00000.csv", std::ifstream::in);

    int total_event_num = std::count(std::istreambuf_iterator<char>(events_str),
                                     std::istreambuf_iterator<char>(), '\n');
    events_str.clear();
    events_str.seekg(0);
    std::string line;
    std::vector<double> t, x, y;
    int event_num = 0;
    int64_t middle_t=-1;
    for (int i = 0; i < total_event_num; i++)
    {
        std::getline(events_str, line);
        std::stringstream lineStream(line);
        std::string cell;
        std::getline(lineStream, cell, ',');
        int64_t time = stoll(cell);
        if(middle_t<0){
            middle_t=time+ 5 * 1e6;;
        }
        t.push_back((time-middle_t)/1e9);
        std::getline(lineStream, cell, ',');
        x.push_back(stod(cell));
        std::getline(lineStream, cell, ',');
        y.push_back(stod(cell));
        event_num++;
    }

    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    // double x[2] = {66, 77};
    // double y[2] = {55, 99};
    // double t[2] = {0.0, 0.005};
    // const double initial_rotations[3] = {0.0000001, 0.0000001, 0.0000001};
    // const double initial_rotations[3] = {1e-3, 1e-3, 1e-3};
    const double initial_rotations[3] = {1, 1, 1};
    // const double initial_rotations[3] = {1.034271551346297, 1.737211928725288, -5.752976192620636};
    double rotations[3];

    // manual diff
    // Build the problem.
    double total_time_ms = 0;

    

    McGradient *mc_gr = new McGradient(fx, fy, cx, cy, height, width);
    mc_gr->ReplaceData( x, y,t, event_num);
    // McGradientBilinear *mc_gr = new McGradientBilinear(fx, fy, cx, cy, x, y, t, height, width, event_num, true);
    int granularity = 300;

    rotations[0] == 0;
    std::ofstream outfile("plot_yz_con.csv", std::ios::out);
    std::ofstream outfile_x("plot_yz_con_x.csv", std::ios::out);
    std::ofstream outfile_y("plot_yz_con_y.csv", std::ios::out);
    std::ofstream outfile_z("plot_yz_con_z.csv", std::ios::out);
    for (int theta_y = 0; theta_y <= granularity; theta_y ++)
    {
        std::cout << "processing " << theta_y << std::endl;
        for (int theta_z = 0; theta_z <= granularity; theta_z ++)
        {

        // std::cout << "processing " << theta_z << std::endl;
            rotations[1] = (theta_y-granularity/2.0)*2*35.0/granularity;

            rotations[2] = (theta_z-granularity/2.0)*2*35.0/granularity;
            double contrast[1];
            double jacobians[3];
            mc_gr->Evaluate(rotations, contrast, jacobians);
            outfile << rotations[1]<<","<<rotations[2]<<","<<-contrast[0]<< std::endl;
            outfile_x << rotations[1]<<","<<rotations[2]<<","<<-jacobians[0]<< std::endl;
            outfile_y << rotations[1]<<","<<rotations[2]<<","<<-jacobians[1]<< std::endl;
            outfile_z << rotations[1]<<","<<rotations[2]<<","<<-jacobians[2]<< std::endl;
        }
    }
    outfile.close();
    outfile_x.close();
    outfile_y.close();
    outfile_z.close();

    
    rotations[2] == 0;
    std::ofstream outfile_2("plot_xy_con.csv", std::ios::out);
    std::ofstream outfile_2x("plot_xy_con_x.csv", std::ios::out);
    std::ofstream outfile_2y("plot_xy_con_y.csv", std::ios::out);
    std::ofstream outfile_2z("plot_xy_con_z.csv", std::ios::out);
    for (int theta_y = 0; theta_y <= granularity; theta_y ++)
    {
        std::cout << "processing " << theta_y << std::endl;
        for (int theta_x = 0; theta_x <= granularity; theta_x ++)
        {

        // std::cout << "processing " << theta_z << std::endl;
            rotations[1] = (theta_y-granularity/2)*2*35.0/granularity;

            rotations[0] = (theta_x-granularity/2)*2*35.0/granularity;
            double contrast[1];
            double jacobians[3];
            mc_gr->Evaluate(rotations, contrast, jacobians);
            outfile_2 << rotations[0]<<","<<rotations[1]<<","<<-contrast[0]<< std::endl;
            outfile_2x << rotations[0]<<","<<rotations[1]<<","<<-jacobians[0]<< std::endl;
            outfile_2y << rotations[0]<<","<<rotations[1]<<","<<-jacobians[1]<< std::endl;
            outfile_2z << rotations[0]<<","<<rotations[1]<<","<<-jacobians[2]<< std::endl;
        }
    }
    outfile_2.close();
    outfile_2x.close();
    outfile_2y.close();
    outfile_2z.close();
    return 0;
}