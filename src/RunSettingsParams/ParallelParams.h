/*
 * ParallelParams.h
 *
 *  Created on: Feb 4, 2024
 *      Author: scott
 */

#ifndef PARALLEL_PARAMS_H
#define PARALLEL_PARAMS_H

#include <array>
#include "RunEval/RunData.h"

/**
 * @brief Abstract class for holding and processing parallelization parameters.
 * Child class(es) specific to implementation(s) must be defined.
 * 
 */
class ParallelParams {
public:
  /**
   * @brief Set parallel parameters for each kernel to the same input dimensions
   * 
   * @param parallelDims 
   */
  virtual void SetParallelDims(const std::array<unsigned int, 2>& parallelDims) = 0;

  /**
   * @brief Retrieve current parallel parameters as RunData type
   * 
   * @return RunData 
   */
  virtual RunData AsRunData() const = 0;

  /**
   * @brief Add results from run with same specified parallel parameters
   * used every parallel component
   * 
   * @param p_params_curr_run 
   * @param curr_run_data 
   */
  virtual void AddTestResultsForParallelParams(
    const std::array<unsigned int, 2>& p_params_curr_run,
    const RunData& curr_run_data) = 0;

  /**
   * @brief Retrieve optimized parameters from results across multiple runs with different
   * parallel parameters and set current parameters to retrieved optimized parameters
   * 
   */
  virtual void SetOptimizedParams() = 0;

  /**
   * @brief Get optimized parallel parameters for parallel processing kernel for
   * kernel that is indexed as an array of two unsigned integers
   * 
   * @param kernel_location 
   * @return std::array<unsigned int, 2> 
   */
  virtual std::array<unsigned int, 2> OptParamsForKernel(
    const std::array<unsigned int, 2>& kernel_location) const = 0;
};

#endif //PARALLEL_PARAMS_H
