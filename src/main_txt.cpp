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
#include "motion_compensation.h"
// #include "reduce.h"
// #include <cblas.h>
// #include "mc_functor.h"
// #include "mc_gradient.h"
#include "mc_gradient_bilinear.h"
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

    std::cout.precision(std::numeric_limits<float>::digits10 + 1);
    int height = 180;
    int width = 240;
    // int height = 480;
    // int width = 640;
    float lower_bound = -5 * 2 * M_PI;
    float upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    float fx = 199.092366542, fy = 198.82882047, cx = 132.192071378, cy = 110.712660011; // boxes
    assert(argc>1);
    google::InitGoogleLogging(argv[0]);

    // load csv to x,y,t
    // std::ifstream events_str("boxes_rotation.csv", std::ifstream::in);
    std::ifstream events_str(argv[1], std::ifstream::in);

    int total_event_num = std::count(std::istreambuf_iterator<char>(events_str),
                                     std::istreambuf_iterator<char>(), '\n');
    events_str.clear();
    events_str.seekg(0);
    std::string line;
    std::vector<float>  x, y;
    std::vector<double> t;
    int event_num = 0;
    double prev_time=-1;
    

    
    // const double initial_rotations[3] = {1e-3, 1e-3, 1e-3};
    const double initial_rotations[3] = {1, 1, 1};
    double rotations[3];

    ceres::LineSearchDirectionType line_search_direction_type = ceres::LBFGS;
    ceres::LineSearchType line_search_type = ceres::WOLFE;
    ceres::NonlinearConjugateGradientType nonlinear_conjugate_gradient_type = ceres::FLETCHER_REEVES;
    uint8_t *output_image;
    cudaAllocMapped(&output_image, height * width * sizeof(uint8_t)*2);
    float* containers[13];
    int current_max_event_size=0;
    std::vector<cudaStream_t> streams(2);
    

    // cudaStreamCreate(&streams[0]);
    // cudaStreamCreate(&streams[1]);
    // checkCudaErrors(cudaMalloc(&containers[0], (height) * (width) * sizeof(float) * 4));
    // checkCudaErrors(cudaMallocHost((void **)&containers[1], 128 * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void **)&containers[2], 128 * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void **)&containers[3], 128 * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void **)&containers[4], 128 * sizeof(float)));
    // checkCudaErrors(cudaMallocHost(&containers[5], 4 * sizeof(float)));

    // cudaMemsetAsync(containers[0], 0, (height) * (width) * sizeof(float) * 4);
    // cudaMemsetAsync(containers[1], 0, 128 * sizeof(float));
    // cudaMemsetAsync(containers[2], 0, 128 * sizeof(float));
    // cudaMemsetAsync(containers[3], 0, 128 * sizeof(float));
    // cudaMemsetAsync(containers[4], 0, 128 * sizeof(float));
    
    McGradientBilinear mc_gr_orig(fx, fy, cx, cy, height, width);  
    mc_gr_orig.allocate();
    mc_gr_orig.exportPtrs(containers,streams,current_max_event_size);
    
    float total_time=0;
    float total_time_sq=0;
    int frame_no=0;
    for (int i = 0; i < total_event_num; i++)
    {
        std::getline(events_str, line);
        std::stringstream lineStream(line);
        std::string cell;
        std::getline(lineStream, cell, ' ');
        double time = stod(cell);
        // if(time<30){
        //     continue;
        // }
        if(prev_time<0){
            prev_time=time;
        }
        t.push_back(time);
        std::getline(lineStream, cell, ' ');
        x.push_back(stod(cell));
        std::getline(lineStream, cell, ' ');
        y.push_back(stod(cell));
        event_num++;
        // std::cout << t.back()<< " "<<x.back()<< " "<< y.back()<<std::endl;
        //10ms has passed
        if(time-prev_time>=1e-2){
            
            // McGradientBilinear *mc_gr = new McGradientBilinear(fx, fy, cx, cy, height, width,containers,streams,current_max_event_size);  
            McGradientBilinear *mc_gr = new McGradientBilinear(fx, fy, cx, cy, height, width);  
            ceres::GradientProblemSolver::Options options;
            // STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
            options.line_search_direction_type = line_search_direction_type;
            //  ARMIJO and WOLFE
            options.line_search_type = line_search_type;
            // FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
            options.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type;
            options.max_num_line_search_step_size_iterations = 10;
            // options.function_tolerance = 1e-5;
            options.function_tolerance = 1e-4;
            options.parameter_tolerance = 1e-6;
            options.minimizer_progress_to_stdout = true;

            prev_time=time;
            nvtx3::scoped_range r{"optimization"};
            if(abs(rotations[0])+abs(rotations[1])+abs(rotations[2])<1e-3){

                std::copy(initial_rotations, initial_rotations + 3, rotations);
            }

            mc_gr->ReplaceData(x,y,t,event_num);
            ceres::GradientProblemSolver::Summary summary;
            
            ceres::GradientProblem problem(mc_gr);
            ceres::Solve(options, problem, rotations, &summary);
            // double params[3]={1,1,1};
            // double residual[1]={0};
            // double gradient[3]={0,0,0};
            // mc_gr->Evaluate(params,residual,gradient);
            // std::fill_n(params,3,1);
            // mc_gr->Evaluate(params,residual,gradient);
            // std::fill_n(params,3,0.9999673);
            // mc_gr->Evaluate(params,residual,gradient);
            // std::fill_n(params,3,0.9999673);
            // mc_gr->Evaluate(params,residual,gradient);
            // std::fill_n(params,3,1);
            // mc_gr->Evaluate(params,residual,gradient);
            // std::fill_n(params,3,1);
            // mc_gr->Evaluate(params,residual,gradient);

            std::cout << summary.FullReport() << "\n";
            std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
                      << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
                      << "\n";
            float contrast, contrast_un;
            mc_gr->GenerateImageBilinear(rotations, output_image,contrast);
            mc_gr->GenerateUncompensatedImageBilinear(rotations, output_image+height*width,contrast_un);
            
            std::stringstream run_name;
            // run_name << std::fixed;
            // run_name << std::setprecision(2);
            
            cv::Mat mat(height*2, width, CV_8U, output_image);
            std::stringstream contrast_text;
            // contrast_text<<"5x5 "<<contrast;
            contrast_text<<"7x7 "<<contrast;
            float time_ms=summary.total_time_in_seconds*1000;
            run_name << "images/txt "<<std::setfill('0') << std::setw(5)<<frame_no<<std::setw(0)<<" " << time_ms<<" time_ms "<<contrast << " bilinear contrast.png";
            std::cout << run_name.str() << std::endl;

            std::stringstream contrast_text_un;
            contrast_text_un<<"uncompensated "<<contrast_un;

            cv::putText(mat,contrast_text.str(),cv::Point(0,20),cv::FONT_HERSHEY_TRIPLEX,0.5,cv::Scalar(255));
            cv::putText(mat,contrast_text_un.str(),cv::Point(0,height+20),cv::FONT_HERSHEY_TRIPLEX,0.5,cv::Scalar(255));
            cv::imwrite(run_name.str(), mat);
            t.clear();
            x.clear();
            y.clear();
            event_num=0;
            frame_no++;
            mc_gr->exportPtrs(containers,streams,current_max_event_size);
            total_time+=time_ms;

            // if(frame_no>20)
            // break;
        }
        
    }
    std::cout<<"mean optimization time(ms): "<< total_time/(frame_no+1)<<std::endl;
    return 0;
}