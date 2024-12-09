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
 * @file DriverCudaBp.cpp
 * @author Scott Grauer-Gray
 * @brief This file contains the "main" function that drives the CUDA BP implementation
 * 
 * @copyright Copyright (c) 2024
 */

#include "RunImp/RunImpMultTypesAccels.h"
#include "RunImpCUDA/RunCUDASettings.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "BpRunProcessing/BpSettings.h"
#include "BpRunEvalImp/RunImpMultInputsBp.h"
#include "BpResultsEvaluation/EvaluateImpResultsBp.h"

int main(int argc, char** argv)
{
  //initialize run implementation settings
  run_environment::RunImpSettings run_imp_settings;

  //enable optimization of parallel parameters with setting to use the allow different thread block dimensions
  //on kernels in same run
  //testing on has found that using different parallel parameters (corresponding to thread block dimensions)
  //in different kernels in the optimized CUDA implementation can decrease runtime
  run_imp_settings.opt_parallel_params_setting =
    {true,
     run_environment::OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun};

  //set datatype(s) to use in run processing in evaluation
  run_imp_settings.datatypes_eval_sizes =
    {run_eval::kDataTypesEvalSizes.begin(),
     run_eval::kDataTypesEvalSizes.end()};

  //set default parallel parameters and parallel parameters to benchmark when searching for optimal
  //parallel parameters
  run_imp_settings.p_params_default_opt_settings =
    {run_cuda::kParallelParamsDefault,
     run_cuda::kParallelParameterOptions};

  //set setting of whether or not to use templated loop iterations in implementation 
  run_imp_settings.templated_iters_setting =
    run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated;

  //set path of baseline runtimes and baseline description
  run_imp_settings.baseline_runtimes_path_desc =
    {beliefprop::kBaselineRunDataPath,
     beliefprop::kBaselineRunDesc};

  //set data subsets to evaluate separate from all data
  run_imp_settings.subset_desc_input_sig = beliefprop::kEvalDataSubsets;

  //set run name to first argument if it exists
  //otherwise set to "CurrentRun"
  run_imp_settings.run_name = (argc > 1) ? argv[1] : "CurrentRun";

  //run and evaluate benchmark with multiple inputs and configurations using CUDA acceleration
  RunImpMultTypesAccels().operator()(
    {{run_environment::AccSetting::kCUDA,
      std::make_shared<RunImpMultInputsBp>(run_environment::AccSetting::kCUDA)}},
    run_imp_settings,
    std::make_unique<EvaluateImpResultsBp>());

  return 0;
}
