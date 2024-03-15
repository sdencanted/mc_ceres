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
#include "mc_functor.h"
#include "mc_gradient.h"
#include "mc_leastsquares.h"

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
    int height = 180;
    int width = 240;
    double lower_bound = -5 * 2 * M_PI;
    double upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    double fx=199.092366542, fy=198.82882047, cx=132.192071378, cy=110.712660011;//boxes    
    google::InitGoogleLogging(argv[0]);

    // load csv to x,y,t
    // std::ifstream events_str("boxes_rotation.csv", std::ifstream::in);
    std::ifstream events_str("event.csv", std::ifstream::in);

    int total_event_num = std::count(std::istreambuf_iterator<char>(events_str),
                                     std::istreambuf_iterator<char>(), '\n');
    events_str.clear();
    events_str.seekg(0);
    std::string line;
    std::vector<double> t, x, y;
    int event_num = 0;
    for (int i = 0; i < total_event_num; i++)
    {
        std::getline(events_str, line);
        std::stringstream lineStream(line);
        std::string cell;
        std::getline(lineStream, cell, ',');
        double time = stod(cell);
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
    // double x[2] = {66, 77};
    // double y[2] = {55, 99};
    // double t[2] = {0.0, 0.005};
    // const double initial_rotations[3] = {0.0000001, 0.0000001, 0.0000001};
    const double initial_rotations[3] = {1e-5, 1e-5, 1e-5};
    // const double initial_rotations[3] = {1.034271551346297, 1.737211928725288, -5.752976192620636};
    double rotations[3];
    std::copy(initial_rotations, initial_rotations + 3, rotations);
    assert(rotations[2] == initial_rotations[2]);

    // auto diff
    // Build the problem.
    // ceres::Problem problem;
    // ceres::Solver::Options options;
    // ceres::Solver::Summary summary;
    //   ceres::CostFunction *cost_function =
    //       new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(intrinsics, x_mat, t_vec));

    // numeric diff
    // Build the problem.
    // ceres::Problem problem;
    // ceres::Solver::Options options;
    // ceres::Solver::Summary summary;
    // auto diff_options=ceres::NumericDiffOptions();
    // diff_options.relative_step_size=1e-6;
    //   ceres::CostFunction *cost_function =
    //       new ceres::NumericDiffCostFunction<McCostFunctor, ceres::CENTRAL, 1, 3>(new McCostFunctor(199.092366542, 198.82882047, 132.192071378, 110.712660011, x, y, t, height, width, event_num),ceres::TAKE_OWNERSHIP,3,diff_options);

    // manual diff
    // Build the problem.
    McGradient* mc_gr=new McGradient(fx, fy, cx, cy, x, y, t, height, width, event_num);
    ceres::GradientProblem problem(mc_gr);
    ceres::GradientProblemSolver::Options options;  
    options.max_num_line_search_step_size_iterations=4;
    options.function_tolerance=1e-4;
    options.parameter_tolerance=1e-6;

    ceres::GradientProblemSolver::Summary summary;
    // problem.SetParameterLowerBound(rotations, 0, lower_bound);
    // problem.SetParameterUpperBound(rotations, 0, upper_bound);
    // problem.SetParameterLowerBound(rotations, 1, lower_bound);
    // problem.SetParameterUpperBound(rotations, 1, upper_bound);
    // problem.SetParameterLowerBound(rotations, 2, lower_bound);
    // problem.SetParameterUpperBound(rotations, 2, upper_bound);
    // Run the solver!
    options.minimizer_progress_to_stdout = true;
    ceres::Solve(options, problem, rotations, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
              << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
              << "\n";
    uint8_t *output_image;
    tryCudaAllocMapped(&output_image,height*width*sizeof(uint8_t),"output_image");

    mc_gr->GenerateImage(rotations,output_image,-summary.final_cost);
    cv::Mat mat(height, width, CV_8U,output_image);
    cv::imwrite("output.png", mat);

    return 0;
}