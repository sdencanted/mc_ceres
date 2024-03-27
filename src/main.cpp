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
    // const double initial_rotations[3] = {1, 1, 1};
    const float initial_rotations[3] = {0.9431776, 1.599026, -5.062576};
    double rotations[3];

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
    std::vector<ceres::LineSearchDirectionType> line_search_direction_types{ceres::STEEPEST_DESCENT, ceres::NONLINEAR_CONJUGATE_GRADIENT, ceres::LBFGS, ceres::BFGS};
    std::vector<ceres::LineSearchType> line_search_types{ceres::ARMIJO, ceres::WOLFE};
    std::vector<ceres::NonlinearConjugateGradientType> nonlinear_conjugate_gradient_types{ceres::FLETCHER_REEVES, ceres::POLAK_RIBIERE, ceres::HESTENES_STIEFEL};
    // for (auto line_search_direction_type : line_search_direction_types)
    // {
    //     for (auto line_search_type : line_search_types)
    //     {
    //         for (auto nonlinear_conjugate_gradient_type : nonlinear_conjugate_gradient_types)
    //         {
    int use_middle_ts = 1;
    // for (int use_middle_ts = 0; use_middle_ts < 2; use_middle_ts++)
    // {
    for (int split_func = 1; split_func >=0; split_func--)
    {

        ceres::LineSearchDirectionType line_search_direction_type = ceres::LBFGS;
        ceres::LineSearchType line_search_type = ceres::WOLFE;
        ceres::NonlinearConjugateGradientType nonlinear_conjugate_gradient_type = ceres::FLETCHER_REEVES;
        float total_time_ms = 0;

        std::stringstream run_name;
        run_name << line_search_direction_type << " " << line_search_type << " " << nonlinear_conjugate_gradient_type << " " << (use_middle_ts ? "middle_ts" : "start_ts") << " " << (split_func ? "split" : "merged");
        std::cout << run_name.str() << std::endl;
        for (int i = 0; i < 10; i++)
        {
            std::copy(initial_rotations, initial_rotations + 3, rotations);
            assert(rotations[2] == initial_rotations[2]);

            // McGradient *mc_gr = new McGradient(fx, fy, cx, cy, x, y, t, height, width, event_num, use_middle_ts);
            McGradientBilinear *mc_gr = new McGradientBilinear(fx, fy, cx, cy, x, y, t, height, width, event_num, use_middle_ts, split_func);
            ceres::GradientProblem problem(mc_gr);
            ceres::GradientProblemSolver::Options options;
            // STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
            options.line_search_direction_type = line_search_direction_type;
            //  ARMIJO and WOLFE
            options.line_search_type = line_search_type;
            // FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
            options.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type;
            options.max_num_line_search_step_size_iterations = 4;
            options.function_tolerance = 1e-5;
            options.parameter_tolerance = 1e-6;

            ceres::GradientProblemSolver::Summary summary;
            // problem.SetParameterLowerBound(rotations, 0, lower_bound);
            // problem.SetParameterUpperBound(rotations, 0, upper_bound);
            // problem.SetParameterLowerBound(rotations, 1, lower_bound);
            // problem.SetParameterUpperBound(rotations, 1, upper_bound);
            // problem.SetParameterLowerBound(rotations, 2, lower_bound);
            // problem.SetParameterUpperBound(rotations, 2, upper_bound);
            // Run the solver!
            // options.minimizer_progress_to_stdout = true;
            options.minimizer_progress_to_stdout = i == 9;

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            ceres::Solve(options, problem, rotations, &summary);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_ms;
            cudaEventElapsedTime(&time_ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            if (i >= 5)
                total_time_ms += time_ms;
            if (i == 10-1)
            {
                std::cout << summary.FullReport() << "\n";
                // std::cout << summary.BriefReport() << "\n";

                std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
                          << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
                          << "\n";
                std::cout << "avg time taken over 50 runs: " << total_time_ms / 5 << "ms" << std::endl;
                uint8_t *output_image;
                cudaAllocMapped(&output_image, height * width * sizeof(uint8_t));
                // rotations[0]=0;
                // rotations[1]=0;
                // rotations[2]=0;
                mc_gr->GenerateImage(rotations, output_image);
                cv::Mat mat(height, width, CV_8U, output_image);
                run_name << std::fixed;
                run_name << std::setprecision(2);
                run_name << " " << -summary.final_cost << " contrast " << total_time_ms / 5 << "ms.png";
                cv::imwrite(run_name.str(), mat);
                // }
                //         }
                //     }
                // }
            }
        }
    }

    return 0;
}