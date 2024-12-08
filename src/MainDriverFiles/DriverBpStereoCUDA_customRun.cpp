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
 * @brief 
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
#include "BpRunProcessing/RunBpOnStereoSet.h"
#include "BpOptimizeCUDA/RunBpOnStereoSetCUDA.h"
#include "BpSingleThreadCPU/stereo.h"
//needed to run the implementation a stereo set using CUDA
#include "BpOptimizeCUDA/RunBpOnStereoSetCUDA.h"

int main(int argc, char** argv)
{
  const std::array<std::string, 2> refTestImPath{argv[1], argv[2]};
  beliefprop::BpSettings alg_settings;
  alg_settings.num_disp_vals = std::stoi(argv[3]);
  alg_settings.disc_k_bp = (float)alg_settings.num_disp_vals / 7.5f;
  const unsigned int dispMapScale = 256 / alg_settings.num_disp_vals;

  const auto cudaTBDims = std::array<unsigned int, 2>{32, 4};
  ParallelParamsBp parallel_params{
    run_environment::OptParallelParamsSetting::kSameParallelParamsAllKernels,
    alg_settings.num_levels,
    cudaTBDims};

  std::unique_ptr<RunBpOnStereoSet<float, 0, run_environment::AccSetting::kCUDA>> runOptBpNumItersNoTemplate =
    std::make_unique<RunBpOnStereoSetCUDA<float, 0, run_environment::AccSetting::kCUDA>>();
  const auto run_output = runOptBpNumItersNoTemplate->operator()({refTestImPath[0], refTestImPath[1]}, alg_settings, parallel_params);
  std::cout << "Output disparity map saved to " << argv[4] << std::endl;
  run_output->out_disparity_map.SaveDisparityMap(argv[4], dispMapScale);
  std::cout << "BP processing runtime (GPU): " << run_output->run_time.count() << std::endl;
  if ((argc > 5) && (std::string(argv[5]) == "comp")) {
    std::unique_ptr<RunBpOnStereoSet<float, 64, run_environment::AccSetting::kNone>> runBpStereoSingleThread = 
      std::make_unique<RunBpOnStereoSetSingleThreadCPU<float, 64, run_environment::AccSetting::kNone>>();
    auto run_output_single_thread = runBpStereoSingleThread->operator()({refTestImPath[0], refTestImPath[1]}, alg_settings, parallel_params);
    std::cout << "BP processing runtime (single threaded imp): " << run_output_single_thread->run_time.count() << std::endl;
    const auto outComp = run_output_single_thread->out_disparity_map.OutputComparison(run_output->out_disparity_map, beliefprop::DisparityMapEvaluationParams());
    std::cout << "Difference between resulting disparity maps (no difference expected)" << std::endl;
    std::cout << outComp.AsRunData();
  }

  return 0;
}
