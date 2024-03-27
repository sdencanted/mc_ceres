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
#include "motion_compensation.h"
#include "reduce.h"
// #include "mc_functor.h"
#include "mc_gradient.h"
#include "mc_gradient_bilinear.h"
// #include "mc_leastsquares.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{

    std::cout.precision(std::numeric_limits<float>::digits10 + 1);
    int height = 180;
    int width = 240;
    float lower_bound = -5 * 2 * M_PI;
    float upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    float fx = 199.092366542, fy = 198.82882047, cx = 132.192071378, cy = 110.712660011; // boxes
    google::InitGoogleLogging(argv[0]);

    // load csv to x,y,t
    // std::ifstream events_str("boxes_rotation.csv", std::ifstream::in);
    std::ifstream events_str("event.csv", std::ifstream::in);

    int total_event_num = std::count(std::istreambuf_iterator<char>(events_str),
                                     std::istreambuf_iterator<char>(), '\n');
    events_str.clear();
    events_str.seekg(0);
    std::string line;
    std::vector<float> t, x, y;
    int event_num = 0;
    for (int i = 0; i < total_event_num; i++)
    {
        std::getline(events_str, line);
        std::stringstream lineStream(line);
        std::string cell;
        std::getline(lineStream, cell, ',');
        float time = stod(cell);
        if ((!slice_window) || (time >= 30 && time < 30.01))
        {
            t.push_back(time);
            std::getline(lineStream, cell, ',');
            x.push_back(stod(cell));
            std::getline(lineStream, cell, ',');
            y.push_back(stod(cell));
            event_num++;
        }
        else if (time >= 30.01)
        {
            break;
        }
    }

    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    // float x[2] = {66, 77};
    // float y[2] = {55, 99};
    // float t[2] = {0.0, 0.005};
    // const float initial_rotations[3] = {0.0000001, 0.0000001, 0.0000001};
    // const double initial_rotations[3] = {1e-3, 1e-3, 1e-3};
    const double initial_rotations[3] = {1, 1, 1};
    // const float initial_rotations[3] = {1.034271551346297, 1.737211928725288, -5.752976192620636};
    double rotations[3];

    // manual diff
    // Build the problem.
    float total_time_ms = 0;

    

    // McGradient *mc_gr = new McGradient(fx, fy, cx, cy, x, y, t, height, width, event_num, true);
    McGradientBilinear *mc_gr = new McGradientBilinear(fx, fy, cx, cy, x, y, t, height, width, event_num, true);
    int granularity = 300;

    rotations[0] == 3.17;
    std::ofstream outfile("plot.csv", std::ios::out);
    for (int theta_y = 0; theta_y <= granularity; theta_y ++)
    {
        std::cout << "processing " << theta_y << std::endl;
        for (int theta_z = 0; theta_z <= granularity; theta_z ++)
        {

        // std::cout << "processing " << theta_z << std::endl;
            rotations[1] = (theta_y-granularity/2)*10.0/granularity;

            rotations[2] = (theta_z-granularity/2)*10.0/granularity;
            double contrast[1];
            double jacobians[3];
            mc_gr->Evaluate(rotations, contrast, jacobians);
            outfile << rotations[1]<<","<<rotations[2]<<","<<-jacobians[1]<< std::endl;
        }
    }
    outfile.close();
    return 0;
}