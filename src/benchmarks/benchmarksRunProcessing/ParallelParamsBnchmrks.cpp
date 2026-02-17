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
 * @file ParallelParamsBnchmrks.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include <ranges>
#include "ParallelParamsBnchmrks.h"

//constructor to set parallel parameters with default dimensions for each kernel
ParallelParamsBnchmrks::ParallelParamsBnchmrks(
    run_environment::OptParallelParamsSetting opt_parallel_params_setting,
    const std::array<unsigned int, 2>& default_parallel_dims) : 
    opt_parallel_params_setting_{opt_parallel_params_setting}
{
  SetParallelDims(default_parallel_dims);
}

//set parallel parameters for each kernel to the same input dimensions
void ParallelParamsBnchmrks::SetParallelDims(
  const std::array<unsigned int, 2>& parallel_dims)
{
  parallel_dims_ = parallel_dims;
}

//get current parallel parameters to data as RunData object
RunData ParallelParamsBnchmrks::AsRunData() const
{
  //initialize RunData object
  RunData curr_run_data;

  //add parallel parameters setting
  curr_run_data.AddDataWHeader(
    std::string(run_environment::kPParamsPerKernelSettingHeader),
    std::string((run_environment::kOptPParmsSettingToDesc.at(
      opt_parallel_params_setting_))));

  //add parallel parameters for each kernel
  curr_run_data.AddDataWHeader(std::string("Parallel Parameters"),
    std::to_string(parallel_dims_[0]) + " x " + std::to_string(parallel_dims_[1]));
  return curr_run_data;
}

//add results from run with same specified parallel parameters used every parallel component
void ParallelParamsBnchmrks::AddTestResultsForParallelParams(const std::array<unsigned int, 2>& p_params_curr_run, const RunData& curr_run_data)
{
  //get total runtime
  p_params_to_run_time_bnchmrks_[p_params_curr_run] =
    *curr_run_data.GetDataAsDouble(run_eval::kOptimizedRuntimeHeader);
}

//retrieve optimized parameters from results across multiple runs with different
//parallel parameters and set current parameters to retrieved optimized
//parameters
void ParallelParamsBnchmrks::SetOptimizedParams() {
  //set optimized parallel parameters to parallel parameters
  //that got the lowest runtime
  const auto best_parallel_params = std::ranges::min_element(
    p_params_to_run_time_bnchmrks_,
    {},
    [](const auto& p_params_w_runtime) {
      return p_params_w_runtime.second;
    })->first;
  SetParallelDims(best_parallel_params);
}
