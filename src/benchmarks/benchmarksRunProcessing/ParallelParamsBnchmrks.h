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
 * @file ParallelParamsBnchmrks.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of ParallelParams to store and process
 * parallelization parameters to use in benchmarks runs
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef PARALLEL_PARAMS_BNCHMRKS_H
#define PARALLEL_PARAMS_BNCHMRKS_H

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <string>
#include "RunSettingsParams/ParallelParams.h"
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunData.h"

namespace benchmarks {
  //constant strings for headers and text related to parallel processing
  constexpr std::string_view kAddBnchmrkPDimsHeader{"Addition Benchmark Parallel Dimensions"};
};

/**
 * @brief Child class of ParallelParams to store and process parallelization
 * parameters to use in each benchmarks kernel
 */
class ParallelParamsBnchmrks final : public ParallelParams {
public:
  /**
   * @brief Constructor to set parallel parameters with default dimensions for
   * each kernel
   * 
   * @param opt_parallel_params_setting 
   * @param default_parallel_dims 
   */
  explicit ParallelParamsBnchmrks(
    run_environment::OptParallelParamsSetting opt_parallel_params_setting,
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
    const RunData& curr_run_data) override;

  /**
   * @brief Retrieve optimized parameters from results across multiple runs
   * with different parallel parameters and set current parameters to
   * retrieved optimized parameters
   */
  void SetOptimizedParams() override;

  /**
   * @brief Get optimized parallel parameters for parallel processing kernel
   * specified by index
   * Currently only one kernel so returning optimized parallel params for that
   * kernel
   * 
   * @param kernel_location 
   * @return Optimized parallel parameters for indexed kernel
   */
  std::array<unsigned int, 2> OptParamsForKernel(
    const std::array<unsigned int, 2>& kernel_location) const override
  {
    return parallel_dims_;
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

  /** @brief Stores the current parallel parameters for benchmarks processing */
  std::array<unsigned int, 2> parallel_dims_;
  
  /** @brief Mapping of parallel parameters to runtime for benchmarks processing */
  std::map<std::array<unsigned int, 2>, double> p_params_to_run_time_bnchmrks_;
};

#endif //PARALLEL_PARAMS_BNCHMRKS_H
