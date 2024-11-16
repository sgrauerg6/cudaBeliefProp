/*
 * ParallelParams.h
 *
 *  Created on: Feb 4, 2024
 *      Author: scott
 */

#ifndef PARALLEL_PARAMS_H
#define PARALLEL_PARAMS_H

#include <array>
#include "RunSettingsEval/RunData.h"

//abstract class for holding and processing parallelization parameters
//child class(es) specific to processing implementation(s) must be defined
class ParallelParams {
public:
  //set parallel parameters for each kernel to the same input dimensions
  virtual void setParallelDims(const std::array<unsigned int, 2>& parallelDims) = 0;

  //retrieve current parallel parameters as RunData type
  virtual RunData runData() const = 0;

  //add results from run with same specified parallel parameters used every parallel component
  virtual void addTestResultsForParallelParams(const std::array<unsigned int, 2>& pParamsCurrRun, const RunData& currRunData) = 0;

  //retrieve optimized parameters from results across multiple runs with different parallel parameters and set current parameters
  //to retrieved optimized parameters
  virtual void setOptimizedParams() = 0;

  //get optimized parallel parameters for parallel processing kernel for kernel that is indexed as an array of two unsigned integers
  virtual std::array<unsigned int, 2> getOptParamsForKernel(const std::array<unsigned int, 2>& kernelLocation) const = 0;
};

#endif //PARALLEL_PARAMS_H
