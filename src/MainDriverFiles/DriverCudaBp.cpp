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
 * @brief Contains the main() function that drives the optimized CUDA
 * belief propagation implementation evaluation across multiple input
 * stereo sets and run configurations.
 * 
 * @copyright Copyright (c) 2024
 */

#include "RunImp/RunImpMultTypesAccels.h"
#include "RunImpCUDA/RunCUDASettings.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include "BpRunProcessing/BpSettings.h"
#include "BpRunEvalImp/RunImpMultInputsBp.h"
#include "BpResultsEvaluation/EvaluateImpResultsBp.h"

/**
 * @brief Main() function that drives the optimized CPU
 * belief propgation implementation evaluation across multiple input
 * stereo sets and run configurations.
 * Takes one input argument that corresponds to run name; recommended that run
 * name include architecture that implementation is run on...if no input
 * argument given run name is set to "CurrentRun"
 * 
 * @param argc 
 * @param argv 
 * @return 0 if successful, another number indicating error if not 
 */
int main(int argc, char** argv)
{
  //initialize run implementation settings
  run_environment::RunImpSettings run_imp_settings;
  
  //set datatype(s) to use in run processing in evaluation
  run_imp_settings.datatypes_eval_sizes =
    {run_eval::kDataTypesEvalSizes.cbegin(),
     run_eval::kDataTypesEvalSizes.cend()};

  //set whether or not to run and evaluate alternate optimized implementations
  //in addition to the "fastest" optimized implementation available
  run_imp_settings.run_alt_optimized_imps = run_eval::kRunAltOptimizedImps;

  //set setting of whether or not to use templated loop iterations in implementation
  //in evaluation runs
  run_imp_settings.templated_iters_setting =
    run_eval::kTemplatedItersEvalSettings;
  
  //enable optimization of parallel parameters with setting to use the allow different thread block dimensions
  //on kernels in same run
  //testing on has found that using different parallel parameters (corresponding to thread block dimensions)
  //in different kernels in the optimized CUDA implementation can decrease runtime
  run_imp_settings.opt_parallel_params_setting =
     run_environment::OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun;

  //set default parallel parameters and parallel parameters to benchmark when searching for optimal
  //parallel parameters
  run_imp_settings.p_params_default_alt_options =
    {run_cuda::kParallelParamsDefault,
     run_cuda::kParallelParameterAltOptions};

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
    {{std::make_shared<RunImpMultInputsBp>(run_environment::AccSetting::kCUDA)}},
    run_imp_settings,
    std::make_unique<EvaluateImpResultsBp>(std::string(beliefprop::kBeliefPropDirectoryName)));

  return 0;
}
