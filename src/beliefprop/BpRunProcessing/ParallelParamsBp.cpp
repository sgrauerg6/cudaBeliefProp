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
 * @file ParallelParamsBp.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include <ranges>
#include "BpResultsEvaluation/DetailedTimingBpConsts.h"
#include "BpRunProcessing/BpRunUtils.h"
#include "ParallelParamsBp.h"

//constructor to set parallel parameters with default dimensions for each kernel
ParallelParamsBp::ParallelParamsBp(
    run_environment::OptParallelParamsSetting opt_parallel_params_setting,
    unsigned int num_levels, const std::array<unsigned int, 2>& default_parallel_dims) : 
    opt_parallel_params_setting_{opt_parallel_params_setting_},
    num_levels_{num_levels}
{
  SetParallelDims(default_parallel_dims);
  //set up mapping of parallel parameters to runtime for each kernel at each level and total runtime
  for (unsigned int i=0; i < beliefprop::kNumKernels; i++) {
    //set to vector length for each kernel to corresponding vector length of kernel in parallel_params.parallel_dims_each_kernel_
    p_params_to_run_time_each_kernel_[i] = std::vector<std::map<std::array<unsigned int, 2>, double>>(parallel_dims_each_kernel_[i].size()); 
  }
  p_params_to_run_time_each_kernel_[beliefprop::kNumKernels] = std::vector<std::map<std::array<unsigned int, 2>, double>>(1); 
}

//set parallel parameters for each kernel to the same input dimensions
void ParallelParamsBp::SetParallelDims(const std::array<unsigned int, 2>& parallel_dims) {
  parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages)] =
    {parallel_dims};
  parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel)] =
    std::vector<std::array<unsigned int, 2>>(num_levels_, parallel_dims);
  parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals)] =
    {parallel_dims};
  parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel)] =
    std::vector<std::array<unsigned int, 2>>(num_levels_, parallel_dims);
  parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel)] =
    std::vector<std::array<unsigned int, 2>>(num_levels_, parallel_dims);
  parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp)] =
    {parallel_dims};
}

//get current parallel parameters to data as RunData object
RunData ParallelParamsBp::AsRunData() const
{
  //initialize RunData object
  RunData curr_run_data;

  //show parallel parameters for each kernel1
  curr_run_data.AddDataWHeader(std::string(beliefprop::kBlurImagesPDimsHeader),
    std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages)][0][0]) + " x " +
    std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages)][0][1]));
  curr_run_data.AddDataWHeader(std::string(beliefprop::kInitMValsPDimsHeader),
    std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals)][0][0]) + " x " +
    std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals)][0][1]));
  for (unsigned int level=0; level < parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel)].size(); level++) {
    curr_run_data.AddDataWHeader(
      std::string(beliefprop::kLevelText) + " " + std::to_string(level) + " " + std::string(beliefprop::kDataCostsPDimsHeader),
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel)][level][0]) + " x " +
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel)][level][1]));
  }
  for (unsigned int level=0; level < parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel)].size(); level++) {
    curr_run_data.AddDataWHeader(
      std::string(beliefprop::kLevelText) + " "  + std::to_string(level) + " " + std::string(beliefprop::kBpItersPDimsHeader),
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel)][level][0]) + " x " +
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel)][level][1]));
  }
  for (unsigned int level=0; level < parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel)].size(); level++) {
    curr_run_data.AddDataWHeader(
      std::string(beliefprop::kLevelText) + " " + std::to_string(level) + " " + std::string(beliefprop::kCopyToNextLevelPDimsHeader),
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel)][level][0]) + " x " +
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel)][level][1]));
  }
  curr_run_data.AddDataWHeader(std::string(beliefprop::kCompOutputDispPDimsHeader),
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp)][0][0]) + " x " +
      std::to_string(parallel_dims_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp)][0][1]));

  return curr_run_data;
}

//add results from run with same specified parallel parameters used every parallel component
void ParallelParamsBp::AddTestResultsForParallelParams(const std::array<unsigned int, 2>& p_params_curr_run, const RunData& curr_run_data)
{
  if (opt_parallel_params_setting_ == run_environment::OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun) {
    for (unsigned int level=0; level < num_levels_; level++) {
      p_params_to_run_time_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel)][level][p_params_curr_run] =
        curr_run_data.GetDataAsDouble(
          std::string(beliefprop::kLevelDCostBpTimeCTimeNames[level][0]) + " " + std::string(run_eval::kMedianOfTestRunsDesc)).value();
      p_params_to_run_time_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel)][level][p_params_curr_run] = 
        curr_run_data.GetDataAsDouble(
          std::string(beliefprop::kLevelDCostBpTimeCTimeNames[level][1]) + " " + std::string(run_eval::kMedianOfTestRunsDesc)).value();
      p_params_to_run_time_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel)][level][p_params_curr_run] =
        curr_run_data.GetDataAsDouble(
          std::string(beliefprop::kLevelDCostBpTimeCTimeNames[level][2]) + " " + std::string(run_eval::kMedianOfTestRunsDesc)).value();
    }
    p_params_to_run_time_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages)][0][p_params_curr_run] =
      curr_run_data.GetDataAsDouble(
        std::string(beliefprop::kTimingNames.at(beliefprop::Runtime_Type::kSmoothing)) + " " + std::string(run_eval::kMedianOfTestRunsDesc)).value();
    p_params_to_run_time_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals)][0][p_params_curr_run] =
      curr_run_data.GetDataAsDouble(
        std::string(beliefprop::kTimingNames.at(beliefprop::Runtime_Type::kInitMessagesKernel)) + " " + std::string(run_eval::kMedianOfTestRunsDesc)).value();
    p_params_to_run_time_each_kernel_[static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp)][0][p_params_curr_run] =
      curr_run_data.GetDataAsDouble(
        std::string(beliefprop::kTimingNames.at(beliefprop::Runtime_Type::kOutputDisparity)) + " " + std::string(run_eval::kMedianOfTestRunsDesc)).value();
  }
  //get total runtime
  p_params_to_run_time_each_kernel_[beliefprop::kNumKernels][0][p_params_curr_run] =
    curr_run_data.GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value();
}

//retrieve optimized parameters from results across multiple runs with different parallel parameters and set current parameters
//to retrieved optimized parameters
void ParallelParamsBp::SetOptimizedParams() {
  if (opt_parallel_params_setting_ == run_environment::OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun) {
    for (unsigned int num_kernel_set = 0; num_kernel_set < parallel_dims_each_kernel_.size(); num_kernel_set++) {
      //retrieve and set optimized parallel parameters for final run
      //std::min_element used to retrieve parallel parameters corresponding to lowest runtime from previous runs
      std::ranges::transform(p_params_to_run_time_each_kernel_[num_kernel_set], 
                             parallel_dims_each_kernel_[num_kernel_set].begin(),
                             [](const auto& tDimsToRunTimeCurrLevel) { 
                             return (std::ranges::min_element(tDimsToRunTimeCurrLevel,
                               [](const auto& a, const auto& b) { return a.second < b.second; }))->first; });
    }
  }
  else {
    //set optimized parallel parameters for all kernels to parallel parameters that got the best runtime across all kernels
    //seems like setting different parallel parameters for different kernels on GPU decrease runtime but increases runtime on CPU
    const auto best_parallel_params = std::ranges::min_element(
      p_params_to_run_time_each_kernel_[beliefprop::kNumKernels][0],
      [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    SetParallelDims(best_parallel_params);
  }
}
