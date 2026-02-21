/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file DriverBpStereoCUDA_customRun.cpp
 * @author Scott Grauer-Gray
 * @brief File with main() function to run optimized CUDA belief propagation
 * implementation on CPU on an input stereo set specified using command line
 * arguments.
 * 
 * @copyright Copyright (c) 2024
 */

//This file contains the "main" function that drives the optimized CPU BP implementation

#include <iostream>
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "BpRunEvalImp/RunImpMultInputsBp.h"
#include "BpRunEvalImp/RunImpOnInputBp.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "RunImp/RunImpMultTypesAccels.h"
#include "BpRunProcessing/RunBpImp.h"
#include "BpOptimizeCUDA/RunBpImpCUDA.h"
#include "BpSingleThreadCPU/RunBpImpSingleThreadCPU.h"
//needed to run the implementation a stereo set using CUDA
#include "BpOptimizeCUDA/RunBpImpCUDA.h"

/**
 * @brief main() function to run optimized CUDA belief propagation implementation
 * on GPU on an input stereo set given using the following input arguments:<br>
 * Argument 1: File path of reference image of stereo set (must be PGM type)<br>
 * Argument 2: File path of test image of stereo set (must be PGM type)<br>
 * Argument 3: Number of possible disparity values<br>
 * Argument 4: File path of disparity image that is generated and saved during program run
 * 
 * @param argc 
 * @param argv 
 * @return 0 if successful, another number indicating error if not 
 */
int main(int argc, char** argv)
{
  //get input stereo set, disparity count, and output disparity map file path
  //from input parameters
  const std::array<std::string, 2> refTestImPath{argv[1], argv[2]};
  beliefprop::BpSettings alg_settings;
  alg_settings.num_disp_vals = std::stoi(argv[3]);
  const std::string out_disp_map_path{argv[4]};

  //generate bp algorithm and parallel parameter settings to use in run
  alg_settings.disc_k_bp = (float)alg_settings.num_disp_vals / 7.5f;
  const unsigned int dispMapScale = 256 / alg_settings.num_disp_vals;
  ParallelParamsBp parallel_params{
    run_environment::OptParallelParamsSetting::kSameParallelParamsAllKernels,
    alg_settings.num_levels,
    run_cuda::kParallelParamsDefault};

  //generate optimized CUDA implementation object to run belief propagation on
  //input stereo set
  std::unique_ptr<RunBpImp<float, 0, run_environment::AccSetting::kCUDA>>
    runOptBpNumItersNoTemplate =
      std::make_unique<RunBpImpCUDA<float, 0, run_environment::AccSetting::kCUDA>>();
  
  //run optimized CUDA belief propagation implementation on GPU on
  //stereo set using specified algorithm settings and parallel parameters
  //resulting run output includes runtime and computed disparity map
  const auto run_output = runOptBpNumItersNoTemplate->operator()(
    {refTestImPath[0], refTestImPath[1]},
    alg_settings,
    parallel_params);

  //display implementation run time and save disparity map to specified file
  //path
  std::cout << "BP processing runtime (GPU): " 
            << run_output->run_time.count() << std::endl;
  std::cout << "Output disparity map saved to "
            << out_disp_map_path << std::endl;
  run_output->out_disparity_map.SaveDisparityMap(
    out_disp_map_path,
    dispMapScale);
  
  //commented out code to compare result to single-thread CPU run...currently
  //only works if disparity count known at compile time since the single-thread
  //CPU implementation only works with disparity count given as template value
  /*if ((argc > 5) && (std::string(argv[5]) == "comp")) {
    std::unique_ptr<RunBpImp<float, 64, run_environment::AccSetting::kNone>> runBpStereoSingleThread = 
      std::make_unique<RunBpImpSingleThreadCPU<float, 64, run_environment::AccSetting::kNone>>();
    auto run_output_single_thread = runBpStereoSingleThread->operator()({refTestImPath[0], refTestImPath[1]}, alg_settings, parallel_params);
    std::cout << "BP processing runtime (single threaded imp): " << run_output_single_thread->run_time.count() << std::endl;
    const auto outComp = run_output_single_thread->out_disparity_map.OutputComparison(run_output->out_disparity_map, beliefprop::DisparityMapEvaluationParams());
    std::cout << "Difference between resulting disparity maps (no difference expected)" << std::endl;
    std::cout << outComp.AsRunData();
  }*/

  return 0;
}
