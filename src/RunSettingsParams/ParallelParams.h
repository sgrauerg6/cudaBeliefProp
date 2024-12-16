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
 * @file ParallelParams.h
 * @author Scott Grauer-Gray
 * @brief Declares abstract class for holding and processing parallelization
 * parameters.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef PARALLEL_PARAMS_H
#define PARALLEL_PARAMS_H

#include <array>
#include "RunEval/RunData.h"

/**
 * @brief Abstract class for holding and processing parallelization parameters.
 * Child class(es) specific to implementation(s) must be defined.
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
