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
 * @file ParallelParamsBp.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of ParallelParams to store and process
 * parallelization parameters to use in each BP kernel at each level
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BP_PARALLEL_PARAMS_H
#define BP_PARALLEL_PARAMS_H

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <string>
#include "RunSettingsParams/ParallelParams.h"
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunData.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"

namespace beliefprop {
  //constant strings for headers and text related to parallel processing
  constexpr std::string_view kBlurImagesPDimsHeader{"Blur Images Parallel Dimensions"};
  constexpr std::string_view kInitMValsPDimsHeader{"Init Message Values Parallel Dimensions"};
  constexpr std::string_view kDataCostsPDimsHeader{"Data Costs Parallel Dimensions"};
  constexpr std::string_view kBpItersPDimsHeader{"BP Thread Parallel Dimensions"};
  constexpr std::string_view kCopyToNextLevelPDimsHeader{"Copy Thread Parallel Dimensions"};
  constexpr std::string_view kCompOutputDispPDimsHeader{"Get Output Disparity Parallel Dimensions"};
  constexpr std::string_view kLevelText{"Level"};
};

/**
 * @brief Child class of ParallelParams to store and process parallelization
 * parameters to use in each BP kernel at each level
 */
class ParallelParamsBp final : public ParallelParams {
public:
  /**
   * @brief Constructor to set parallel parameters with default dimensions for
   * each kernel
   * 
   * @param opt_parallel_params_setting 
   * @param num_levels 
   * @param default_parallel_dims 
   */
  explicit ParallelParamsBp(
    run_environment::OptParallelParamsSetting opt_parallel_params_setting,
    unsigned int num_levels,
    const std::array<unsigned int, 2>& default_parallel_dims);

  /**
   * @brief Set parallel parameters for each kernel to the same input dimensions
   * 
   * @param parallel_dims 
   */
  void SetParallelDims(
    const std::array<unsigned int, 2>& parallel_dims) override;

  /**
   * @brief Add results from run with same specified parallel parameters used
   * every parallel component
   * 
   * @param p_params_curr_run 
   * @param curr_run_data 
   */
  void AddTestResultsForParallelParams(
    const std::array<unsigned int, 2>& p_params_curr_run,
    const RunData& curr_run_data);

  /**
   * @brief Retrieve optimized parameters from results across multiple runs
   * with different parallel parameters and set current parameters to
   * retrieved optimized parameters
   */
  void SetOptimizedParams() override;

  /**
   * @brief Get optimized parallel parameters for parallel processing kernel
   * for kernel that is indexed as an array of two unsigned integers that
   * correspond to the kernel name and bp level
   * 
   * @param kernel_location 
   * @return Optimized parallel parameters for indexed kernel
   */
  std::array<unsigned int, 2> OptParamsForKernel(
    const std::array<unsigned int, 2>& kernel_location) const override
  {
    return parallel_dims_each_kernel_[kernel_location[0]][kernel_location[1]];
  }

  /**
   * @brief Retrieve current parallel parameters as RunData object
   * 
   * @return RunData object containing current parallel parameters 
   */
  RunData AsRunData() const override;

private:
  /** @brief Setting to determine whether or not to use same parallel
   *  parameters for all kernels in run or to allow for different
   *  parallel parameters for each kernel */
  const run_environment::OptParallelParamsSetting opt_parallel_params_setting_;
  
  /** @brief Number of levels in bp processing hierarchy */
  const unsigned int num_levels_;

  /** @brief Stores the current parallel parameters for each processing kernel */
  std::array<std::vector<std::array<unsigned int, 2>>, beliefprop::kNumKernels>
  parallel_dims_each_kernel_;
  
  /** @brief Mapping of parallel parameters to runtime for each kernel at each
   *  level and total runtime */
  std::array<std::vector<std::map<std::array<unsigned int, 2>, double>>,
             (beliefprop::kNumKernels + 1)>
  p_params_to_run_time_each_kernel_;
};

#endif //BP_PARALLEL_PARAMS_H
