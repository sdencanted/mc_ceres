#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
// #include "glog/logging.h"

// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_double.h"
// 
// #include <cblas.h>
// #include "mc_functor.h"
// #include "mc_gradient.h"
#include "mc_gradient_double.h"
// #include "mc_leastsquares.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <nvtx3/nvtx3.hpp>

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

    std::cout.precision(std::numeric_limits<double>::digits10 + 1);
    // int height = 180;
    // int width = 240;
    // int height = 480;
    // int width = 640;
    // int height = 400;
    // int width = 400;
    int height = 720;
    int width = 1280;
    double lower_bound = -5 * 2 * M_PI;
    double upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    // double fx = 199.092366542, fy = 198.82882047, cx = 132.192071378, cy = 110.712660011; // boxes
    double fx = 3.22418800e+03, fy = 3.21510040e+03, cx = (8.80357033e+02), cy = (4.17066114e+02); // evk4
    // double fx = 1.7904096255342997e+03, fy = 1.7822557654303025e+03, cx = (3.2002555821529580e+02)-244, cy = (2.3647053629109917e+02)-84; // dvx micro
    google::InitGoogleLogging(argv[0]);

    // load csv to x,y,t
    // std::ifstream events_str("boxes_rotation.csv", std::ifstream::in);
    // std::ifstream events_str("event.csv", std::ifstream::in);
    std::string filename = "bag_00000.csv";
    if (argc > 1)
    {
        filename = argv[1];
    }
    std::ifstream events_str(filename, std::ifstream::in);

    int total_event_num = std::count(std::istreambuf_iterator<char>(events_str),
                                     std::istreambuf_iterator<char>(), '\n');
    events_str.clear();
    events_str.seekg(0);
    std::string line;
    std::vector<double> t, x, y;
    int event_num = 0;

    // for (int i = 0; i < total_event_num; i++)
    // {
    //     std::getline(events_str, line);
    //     std::stringstream lineStream(line);
    //     std::string cell;
    //     std::getline(lineStream, cell, ',');
    //     double time = stod(cell);
    //     if ((!slice_window) || (time >= 30 && time < 30.01))
    //     {
    //         t.push_back(time);
    //         std::getline(lineStream, cell, ',');
    //         x.push_back(stod(cell));
    //         std::getline(lineStream, cell, ',');
    //         y.push_back(stod(cell));
    //         event_num++;
    //     }
    //     else if (time >= 30.01)
    //     {
    //         break;
    //     }
    // }

    int64_t middle_t = -1;
    for (int i = 0; i < total_event_num; i++)
    {
        std::getline(events_str, line);
        std::stringstream lineStream(line);
        std::string cell;
        std::getline(lineStream, cell, ',');
        int64_t time = stoll(cell);
        if (middle_t < 0)
        {
            middle_t = time + 5 * 1e6;
            ;
        }
        t.push_back((time - middle_t) / 1e9);
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
    // const double initial_rotations[3] = {6.30, 1e-3, 1e-3};
    // const double initial_rotations[3] = {1, 1, 1};
    const double initial_rotations[3] = {1, -20, 1};
    // const double initial_rotations[3] = {0.9431776, 1.599026, -5.062576};
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
    // for (int use_middle_ts = 0; use_middle_ts < 2; use_middle_ts++)
    // {

    ceres::LineSearchDirectionType line_search_direction_type = ceres::LBFGS;
    ceres::LineSearchType line_search_type = ceres::WOLFE;
    ceres::NonlinearConjugateGradientType nonlinear_conjugate_gradient_type = ceres::FLETCHER_REEVES;
    double total_time_ms = 0;

    std::stringstream run_name;
    run_name << line_search_direction_type << " " << line_search_type << " " << nonlinear_conjugate_gradient_type;
    std::cout << run_name.str() << std::endl;

    // McGradient *mc_gr = new McGradient(fx, fy, cx, cy, x, y, t, height, width, event_num, use_middle_ts);

    // std::shared_ptr<McGradient> mc_gr = std::make_shared<McGradient>(fx, fy, cx, cy, height, width);
    McGradient *mc_gr = new McGradient(fx, fy, cx, cy, height, width);
    ceres::GradientProblemSolver::Options options;
    // // STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
    // options.line_search_direction_type = line_search_direction_type;
    // //  ARMIJO and WOLFE
    // options.line_search_type = line_search_type;
    // // FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
    // options.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type;
    options.min_line_search_step_size = 1e-99;
    options.max_num_line_search_step_size_iterations = 4;
    options.function_tolerance = 1e-4;
    // options.function_tolerance = 1e-5;
    options.parameter_tolerance = 1e-6;
    options.minimizer_progress_to_stdout = true;
    for (int i = 0; i < 1; i++)
    {
        nvtx3::scoped_range r{"optimization"};
        std::copy(initial_rotations, initial_rotations + 3, rotations);
        assert(rotations[2] == initial_rotations[2]);
        mc_gr->ReplaceData(x, y, t, event_num);
        double residuals[1] = {0};
        double gradient[3] = {0};

        for (int y = 0; y < 10; y++)
        {

            mc_gr->Evaluate(rotations, residuals, gradient);
            std::copy(initial_rotations, initial_rotations + 3, rotations);
            // mc_gr->EvaluateCpu(rotations, residuals, gradient);
            // std::copy(initial_rotations, initial_rotations + 3, rotations);
            // double image_sample[100];
            // cudaMemcpy(image_sample, mc_gr->image_ + width * 200, sizeof(double) * 100, cudaMemcpyDefault);
            // for (int u = 0;u < 10; u++)
            // {
            //     std::cout << image_sample[u] << std::endl;
            // }
        }

        // rotations[1] = -20;
        // McGradientInterface *mc_gr_interface = new McGradientInterface(mc_gr);
        ceres::GradientProblem problem(mc_gr);

        ceres::GradientProblemSolver::Summary summary;
        options.minimizer_progress_to_stdout= true;

        ceres::Solve(options, problem, rotations, &summary);

        cudaDeviceSynchronize();

        if (i == 0)
        {
            std::cout << summary.FullReport() << "\n";
            // std::cout << summary.BriefReport() << "\n";

            std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
                      << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
                      << "\n";
            uint8_t *output_image;
            cudaAllocMapped(&output_image, height * width * sizeof(uint8_t));
            // // rotations[0]=0;
            // // rotations[1]=0;
            // // rotations[2]=0;
            double contrast;
            mc_gr->GenerateImage(rotations, output_image, contrast);
            cv::Mat mat(height, width, CV_8U, output_image);
            run_name << std::fixed;
            run_name << std::setprecision(2);
            run_name << " " << -summary.final_cost << " contrast.png";
            cv::imwrite(run_name.str(), mat);
            // double xprime[100];
            // cudaMemcpy(xprime,mc_gr->x_prime_,100*sizeof(double),cudaMemcpyDefault);
            // for(int u=0;u<100;u++){
            //     std::cout<<xprime[u]<<std::endl;
            // }
            // }
            //         }
            //     }
            // }
        }
    }

    return 0;
}