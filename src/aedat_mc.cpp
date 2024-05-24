#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
// #include "glog/logging.h"
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/core/multi_stream_slicer.hpp>
// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_float.h"
// 
// #include <cblas.h>
// #include "mc_functor.h"
// #include "mc_gradient.h"
#include "mc_gradient.h"
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
    using namespace std::chrono_literals;
    bool integrate_reduction = argc > 1;

    std::cout.precision(std::numeric_limits<float>::digits10 + 1);
    int height = 480;
    int width = 640;
    float lower_bound = -5 * 2 * M_PI;
    float upper_bound = 5 * 2 * M_PI;
    bool slice_window = false;
    float fx = 1.7904096255342997e+03, fy = 1.7822557654303025e+03, cx = 3.2002555821529580e+02, cy = 2.3647053629109917e+02; // boxes
    google::InitGoogleLogging(argv[0]);

    // dv::io::MonoCameraRecording reader("1hz_mini_lowest_bias_roi_x_300_339-2024_01_25_14_29_17.aedat4");
    std::shared_ptr<dv::io::MonoCameraRecording> reader;
    if (argc < 2)
    {
        reader = std::make_shared<dv::io::MonoCameraRecording>("/home/airlab/fan-2024_04_17_17_46_40.aedat4");
    }
    else
    {

        reader = std::make_shared<dv::io::MonoCameraRecording>(argv[1]);
    }
    std::string eventStream;

    // Find streams with compatible types from the list of all available streams
    for (const auto &name : reader->getStreamNames())
    {
        if (reader->isStreamOfDataType<dv::EventPacket>(name) && eventStream.empty())
        {
            eventStream = name;
        }
    }

    assert(!eventStream.empty());

    // const float initial_rotations[3] = {0.9431776, 1.599026, -5.062576};
    // const float initial_rotations[3] = {10, 10, 10};
    const float initial_rotations[3] = {1, 1, 1};
    double rotations[3];

    std::copy(initial_rotations, initial_rotations + 3, rotations);
    ceres::LineSearchDirectionType line_search_direction_type = ceres::LBFGS;
    ceres::LineSearchType line_search_type = ceres::WOLFE;
    ceres::NonlinearConjugateGradientType nonlinear_conjugate_gradient_type = ceres::FLETCHER_REEVES;
    float total_time_ms = 0;

    ceres::GradientProblemSolver::Options options;
    // STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
    options.line_search_direction_type = line_search_direction_type;
    //  ARMIJO and WOLFE
    options.line_search_type = line_search_type;
    // FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
    options.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type;
    options.max_num_line_search_step_size_iterations = 4;
    options.function_tolerance = 1e-5;
    // options.parameter_tolerance = 1e-6;
    options.parameter_tolerance = 1e-6;
    options.minimizer_progress_to_stdout = false;
    std::shared_ptr<McGradient> mc_gr = std::make_shared<McGradient>(fx, fy, cx, cy, height, width);

    // Slice the data every 10 milliseconds

    uint8_t *output_image;
    cudaMallocHost(&output_image, height * width * sizeof(uint8_t) * 2);
    int packet_count = 0;

    ceres::GradientProblemSolver::Summary summary;
    dv::EventStreamSlicer slicer;
    ceres::GradientProblem problem(mc_gr.get());
    slicer.doEveryTimeInterval(10ms, [&](const dv::AddressableEventStorage<dv::Event, dv::EventPacket> &data)
                               {
                                
                                    mc_gr->ReplaceData(data);
                                    
                                    // McGradientInterface *mc_gr_interface = new McGradientInterface(mc_gr);

                                    if(abs(rotations[0])+abs(rotations[1])+abs(rotations[2])<1e-3){

                                        std::copy(initial_rotations, initial_rotations + 3, rotations);
                                    }
                                    nvtx3::scoped_range r{"optimization"};
                                    double residuals[1];
                                    double gradients[3];
                                   ceres::Solve(options, problem, rotations, &summary);
                                   cudaDeviceSynchronize();
                                   if(packet_count%1==0){
                                        //  std::cout << summary.FullReport() << "\n";

                                        std::cout << "rot : " << initial_rotations[0] << " " << initial_rotations[1] << " " << initial_rotations[2] << " "
                                                << " -> " << rotations[0] << " " << rotations[1] << " " << rotations[2] << " "
                                                << "\n";
                                        float contrast;
                                        mc_gr->GenerateImage(rotations, output_image,contrast);
                                        mc_gr->GenerateUncompensatedImage(rotations, output_image+height*width,contrast);
                                        cv::Mat mat(height*2, width, CV_8U, output_image);
                                        
                                        std::stringstream run_name;
                                        run_name << std::fixed;
                                        run_name << std::setprecision(3);
                                        run_name << "images/1hz "<<packet_count<<" " << -summary.final_cost<<" "<<rotations[0]<<" "<<rotations[1]<<" "<<rotations[2] << " contrast.png";
                                        cv::imwrite(run_name.str(),mat);
                                   }

                                    packet_count++; 
                                    });

    // Read event in a loop, this is needed since events are stored in small batches of short period of time
    while (const auto events = reader->getNextEventBatch(eventStream))
    {
        // std::cout << events->getHighestTime() << std::endl;
        // Pass events to the slicer
        slicer.accept(*events);
    }

    cudaFreeHost(output_image);

    return 0;
}