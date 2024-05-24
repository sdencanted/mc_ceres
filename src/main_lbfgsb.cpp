

// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_float.h"
#include "mc_gradient_lbfgspp.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <nvtx3/nvtx3.hpp>

using namespace LBFGSpp;
using Eigen::VectorXf;
int main(int argc, char **argv)
{
    
    // cv::Mat mat2;
    // uint8_t output_image1[height*width];
    // cv::Mat mat1(height, width, CV_8U, output_image1);
    // openblas_set_num_threads(1);
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    // cudaSetDeviceFlags(cudaDeviceScheduleYield);
    // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    bool integrate_reduction = argc > 1;

    std::cout.precision(std::numeric_limits<float>::digits10 + 1);
    // int height = 180;
    // int width = 240;
    // int height = 480;
    // int width = 640;
    // int height = 400;
    // int width = 400;
    int height = 720;
    int width = 1280;
    float lower_bound = -5 * 2 * M_PI;
    float upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    // float fx = 199.092366542, fy = 198.82882047, cx = 132.192071378, cy = 110.712660011; // boxes
    float fx = 3.22418800e+03, fy = 3.21510040e+03, cx = (8.80357033e+02), cy = (4.17066114e+02) ; // evk4
    // float fx = 1.7904096255342997e+03, fy = 1.7822557654303025e+03, cx = (3.2002555821529580e+02)-244, cy = (2.3647053629109917e+02)-84; // dvx micro
    

    // load csv to x,y,t
    // std::ifstream events_str("boxes_rotation.csv", std::ifstream::in);
    // std::ifstream events_str("event.csv", std::ifstream::in);
    std::string filename="bag_00000.csv";
    if(argc>1){
        filename=argv[1];
    }
    std::ifstream events_str(filename, std::ifstream::in);

    int total_event_num = std::count(std::istreambuf_iterator<char>(events_str),
                                     std::istreambuf_iterator<char>(), '\n');
    events_str.clear();
    events_str.seekg(0);
    std::string line;
    std::vector<float> t, x, y;
    int event_num = 0;
    

    
    int64_t middle_t=-1;



    
    // Set up parameters
    LBFGSBParam<float> param;  // New parameter class
    param.m=10;
    param.epsilon = 1e-2;
    // param.epsilon_rel = 1e-3;
    param.max_iterations = 100;
    param.delta = 1e-4;
    
    // c1 aka sufficient decrease
    param.ftol = 1e-6;

    // c2 aka curvature
    param.wolfe = 0.99;

    // Create solver and function object
    LBFGSBSolver<float> solver(param);  // New solver class


    // Bounds
    Eigen::Vector3f lb = {-M_PI,-6.0*2*M_PI,-M_PI};
    Eigen::Vector3f ub = {M_PI,0,M_PI};

    // Initial guess
    Eigen::VectorXf rotations=Eigen::VectorXf::Constant(3,1e-3); 
    rotations[1]=-1;

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
    // const float initial_rotations[3] = {0.0000001, 0.0000001, 0.0000001};
    // const double initial_rotations[3] = {1e-3, 1e-3, 1e-3};
    // const double initial_rotations[3] = {6.30, 1e-3, 1e-3};
    // double initial_rotations[3] = {1, 1, 1};
    // if(argc>=5){
    //     initial_rotations[0]=std::stof(argv[2]);
    //     initial_rotations[1]=std::stof(argv[3]);
    //     initial_rotations[2]=std::stof(argv[4]);
    // }
    // const float initial_rotations[3] = {0.9431776, 1.599026, -5.062576};
    // double rotations[3];

    float total_time_ms = 0;


    McGradient* mc_gr = new McGradient(fx, fy, cx, cy, height, width);
    for (int i = 0; i < 1; i++)
    {
        
        mc_gr->ReplaceData(x,y,t,event_num);
            // x will be overwritten to be the best point found
            float fx;
        int niter = solver.minimize(*mc_gr, rotations, fx, lb, ub);

        std::cout << mc_gr->iterations << " iterations" << std::endl;
        std::cout << "x = \n" << rotations.transpose() << std::endl;
        std::cout << "f(x) = " << fx << std::endl;
        

        // nvtx3::scoped_range r{"optimization"};
        // std::copy(initial_rotations, initial_rotations + 3, rotations);
        // assert(rotations[2] == initial_rotations[2]);
        // mc_gr->ReplaceData(x,y,t,event_num);
        // double residuals[1]={0};
        // double gradient[3]={0};

        // cudaDeviceSynchronize();
        
        // if (i ==0)
        // {
        //     std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
        //               << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
        //               << "\n";
        //     uint8_t *output_image;
        //     cudaAllocMapped(&output_image, height * width * sizeof(uint8_t));
        //     float contrast;
        //     mc_gr->GenerateImage(rotations, output_image,contrast);
        //     cv::Mat mat(height, width, CV_8U, output_image);
        //     run_name << std::fixed;
        //     run_name << std::setprecision(2);
        //     run_name << " " << -summary.final_cost << " contrast.png";
        //     cv::imwrite(run_name.str(), mat);
        // }
    }

    return 0;
}