// #include "ceres/ceres.h"
// #include "ceres/numeric_diff_options.h"
// #include "glog/logging.h"
#include <algorithm>
#include <chrono>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
// #include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation.h"
// #include "reduce.h"
// #include "mc_functor.h"
// #include "mc_gradient.h"
#include "mc_gradient_bilinear_gsl.h"
// #include "mc_leastsquares.h"

#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>
#include <nvtx3/nvtx3.hpp>
void local_contrast_fdf(const gsl_vector *v, void *ptr, double *f, gsl_vector *df)
{
    McGradientBilinear *mc_gr = (McGradientBilinear *)ptr;
    // McGradient *mc_gr = (McGradient*)ptr;

    std::cout
            << v->data[0]<< " "
            << v->data[1] << " "
            << v->data[2] <<std::endl;
    if (df != nullptr)
    {
        printf("fdf\n");
        mc_gr->f_count++;
        mc_gr->g_count++;
        mc_gr->Evaluate(v->data, f, df->data);
    }
    else
    {
        printf("f\n");
        mc_gr->f_count++;
        mc_gr->Evaluate(v->data, f, nullptr);
    }
}
double local_contrast_f(const gsl_vector *v, void *adata)
{
    double cost;
    local_contrast_fdf(v, adata, &cost, nullptr);
    return cost;
}

void local_contrast_df(const gsl_vector *v, void *adata, gsl_vector *df)
{
    double cost;
    local_contrast_fdf(v, adata, &cost, df);
}
int main(int argc, char **argv)
{
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    // cudaSetDeviceFlags(cudaDeviceScheduleYield);
    // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    {

        std::cout.precision(std::numeric_limits<float>::digits10 + 1);
        int height = 180;
        int width = 240;
        float lower_bound = -5 * 2 * M_PI;
        float upper_bound = 5 * 2 * M_PI;
        bool slice_window = false;
        float fx = 199.092366542, fy = 198.82882047, cx = 132.192071378, cy = 110.712660011; // boxes
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
        // const double initial_rotations[3] = {0.9431776, 1.599026, -5.062576};
        // const float initial_rotations[3] = {1.034271551346297, 1.737211928725288, -5.752976192620636};
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
        
        uint8_t output_image[height*width];
        cv::Mat mat(height, width, CV_8U, output_image);
        for (int use_middle_ts = 0; use_middle_ts < 2; use_middle_ts++)
        {

            float total_time_ms = 0;

            std::stringstream run_name;
            run_name << (use_middle_ts ? "middle_ts" : "start_ts");
            std::cout << run_name.str() << std::endl;
            for (int i = 0; i < 10; i++)
            {
                std::copy(initial_rotations, initial_rotations + 3, rotations);
                assert(rotations[2] == initial_rotations[2]);

                // McGradient *mc_gr = new McGradient(fx, fy, cx, cy, x, y, t, height, width, event_num, use_middle_ts);
                McGradientBilinear *mc_gr = new McGradientBilinear(fx, fy, cx, cy, x, y, t, height, width, event_num);

                // PREPARE SOLVER
                //  Choose a solver/minimizer type (algorithm)
                const gsl_multimin_fdfminimizer_type *solver_type;
                // A. Non-linear conjugate gradient
                // solver_type = gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves
                solver_type = gsl_multimin_fdfminimizer_conjugate_pr; // Polak-Ribiere
                //  B. quasi-Newton methods: Broyden-Fletcher-Goldfarb-Shanno
                // solver_type = gsl_multimin_fdfminimizer_vector_bfgs; // BFGS
                // solver_type = gsl_multimin_fdfminimizer_vector_bfgs2; // BFGS2 is not as good

                // Routines to compute the cost function and its derivatives
                gsl_multimin_function_fdf solver_info;

                const int num_params = 3;   // Size of angular velocity
                solver_info.n = num_params; // Size of the parameter vector

                solver_info.f = local_contrast_f;     // Cost function
                solver_info.df = local_contrast_df;   // Gradient of cost function
                solver_info.fdf = local_contrast_fdf; // Cost and gradient functions
                solver_info.params = mc_gr;
                // Initialize solver
                gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc(solver_type, num_params);
                const double initial_step_size = 1.0;

                double tol = 0.1; // for solvers fr, pr and bfgs
                std::string str_bfgs2("vector_bfgs2");
                if (str_bfgs2.compare(solver_type->name) == 0)
                {
                    tol = 0.8;
                }

                // Initial parameter vector
                gsl_vector *vx = gsl_vector_alloc(num_params);
                gsl_vector_set(vx, 0, initial_rotations[0]);
                gsl_vector_set(vx, 1, initial_rotations[1]);
                gsl_vector_set(vx, 2, initial_rotations[2]);

                // This call already evaluates the function
                gsl_multimin_fdfminimizer_set(solver, &solver_info, vx, initial_step_size, tol);

                // const double initial_cost = contrast_f(vx, &oAuxdata);
                const double initial_cost = solver->f;

                // ITERATE
                const int num_max_line_searches = 50;
                //    const int num_max_line_searches = params.optim_params.max_num_iters;
                int status;
                // const double epsabs_grad = 1e-8, tolfun = 1e-5;
                const double tolfun = 1e-5;
                //    const double epsabs_grad = params.optim_params.gradient_tolerance,
                //                 tolfun = params.optim_params.function_tolerance;
                double cost_new = 1e9, cost_old = 1e9;
                size_t iter = 0;

                std::cout << "Optimization. Solver type = " << solver_type->name << std::endl;
                std::cout << "iter=" << std::setw(3) << iter << "  ang_vel=["
                          << gsl_vector_get(solver->x, 0) << " "
                          << gsl_vector_get(solver->x, 1) << " "
                          << gsl_vector_get(solver->x, 2) << "]  cost=" << std::setprecision(std::numeric_limits<float>::digits10 + 1) << solver->f << std::endl;

                nvtxRangePushA("contrast"); // Begins NVTX range
                // solve here
                do
                {
                    iter++;
                    cost_old = cost_new;
                    status = gsl_multimin_fdfminimizer_iterate(solver);
                    // status == GLS_SUCCESS (0) means that the iteration reduced the function value

                    std::cout << "iter=" << std::setw(3) << iter << "  ang_vel=["
                            << gsl_vector_get(solver->x, 0) << " "
                            << gsl_vector_get(solver->x, 1) << " "
                            << gsl_vector_get(solver->x, 2) << "]  cost=" << std::setprecision(std::numeric_limits<float>::digits10 + 1) << solver->f<<std::endl;


                    if (status == GSL_SUCCESS)
                    {
                        // Test convergence due to stagnation in the value of the function
                        cost_new = gsl_multimin_fdfminimizer_minimum(solver);
                        if (fabs(1 - cost_new / (cost_old + 1e-7)) < tolfun)
                        {
                            std::cout << "progress tolerance reached." << std::endl;
                            break;
                        }
                        else
                            status = GSL_CONTINUE;
                    }

                    // // Test convergence due to absolute norm of the gradient
                    // if (GSL_SUCCESS == gsl_multimin_test_gradient(solver->gradient, epsabs_grad))
                    // {
                    //     std::cout << "gradient tolerance reached." << std::endl;
                    //     break;
                    // }

                    if (status != GSL_CONTINUE)
                    {
                        // The iteration was not successful (did not reduce the function value)
                        std::cout << "stopped iteration; status = " << status << std::endl;
                        std::cout << "iteration is not making progress towards solution" << std::endl;
                        break;
                    }
                } while (status == GSL_CONTINUE && iter < num_max_line_searches);
                
                nvtxRangePop();

                nvtxRangePushA("results"); // Begins NVTX range
                // Convert from GSL to OpenCV format
                gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);
                rotations[0] = gsl_vector_get(final_x, 0);
                rotations[1] = gsl_vector_get(final_x, 1);
                rotations[2] = gsl_vector_get(final_x, 2);
                

                // const double final_cost = contrast_f(final_x, &oAuxdata);
                // const double final_cost = solver->f;
                const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

                std::cout << "--- Initial cost = " << std::setprecision(std::numeric_limits<float>::digits10 + 1) << initial_cost << std::endl;
                std::cout << "--- Final cost   = " << std::setprecision(std::numeric_limits<float>::digits10 + 1) << final_cost << std::endl;
                std::cout << "--- iter=" << std::setw(3) << iter << "  ang_vel=["
                          << rotations[0] << " " << rotations[1] << " " << rotations[2] << "]" << std::endl;
                std::cout << "--- function evaluations + gradient evaluations = "
                          << mc_gr->f_count << " + " << mc_gr->g_count << std::endl;

                // nvtxRangePushA("results1"); // Begins NVTX range
                // status = gsl_multimin_fdfminimizer_iterate(solver);
                // nvtxRangePop();

                // Release memory used during optimization
                gsl_multimin_fdfminimizer_free(solver);
                gsl_vector_free(vx);
                
                if (i == 9)
                {
                    // print results here

                    std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
                              << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
                              << "\n";
                    // std::cout << "avg time taken over 10 runs: " << total_time_ms / 10 << "ms" << std::endl;
                    // uint8_t *output_image;
                    uint8_t output_image[height*width];
                    // cudaAllocMapped(&output_image, height * width * sizeof(uint8_t));
                    mc_gr->GenerateImage(rotations, output_image);

                    cv::Mat mat(height, width, CV_8U, output_image);
                    // cudaFreeHost(output_image);
                    run_name << std::fixed;
                    run_name << std::setprecision(2);
                    run_name << " " << -final_cost << " contrast " << total_time_ms / 10 << "ms.png";
                    cv::imwrite(run_name.str(), mat);
                }
                
                nvtxRangePop();
            }
        }
    }

    return 0;
}