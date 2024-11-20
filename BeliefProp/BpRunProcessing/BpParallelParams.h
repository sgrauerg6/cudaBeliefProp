/*
 * BpParallelParams.h
 *
 *  Created on: Feb 4, 2024
 *      Author: scott
 */

#ifndef BP_PARALLEL_PARAMS_H
#define BP_PARALLEL_PARAMS_H

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <string>
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpResultsEvaluation/DetailedTimingBPConsts.h"
#include "BpRunProcessing/BpStructsAndEnums.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunImp/ParallelParams.h"

//structure containing parameters including parallelization parameters
//to use at each BP level
class BpParallelParams final : public ParallelParams {
public:
  //constructor to set parallel parameters with default dimensions for each kernel
  BpParallelParams(run_environment::OptParallelParamsSetting opt_parallel_params_setting,
    unsigned int num_levels, const std::array<unsigned int, 2>& default_parallel_dims);

  //set parallel parameters for each kernel to the same input dimensions
  void SetParallelDims(const std::array<unsigned int, 2>& parallel_dims) override;

  //add results from run with same specified parallel parameters used every parallel component
  void AddTestResultsForParallelParams(const std::array<unsigned int, 2>& p_params_curr_run, const RunData& curr_run_data);

  //retrieve optimized parameters from results across multiple runs with different parallel parameters and set current parameters
  //to retrieved optimized parameters
  void SetOptimizedParams() override;

  //get optimized parallel parameters for parallel processing kernel for kernel that is indexed as an array of two unsigned integers
  //that correspond to the kernel name and bp level
  std::array<unsigned int, 2> OptParamsForKernel(const std::array<unsigned int, 2>& kernel_location) const override {
    return parallel_dims_each_kernel_[kernel_location[0]][kernel_location[1]];
  }

  //retrieve current parallel parameters as RunData object
  RunData AsRunData() const override;

private:
  //setting to determine whether or not to use same parallel parameters for all kernels in run or to allow for different
  //parallel parameters for each kernel
  const run_environment::OptParallelParamsSetting opt_parallel_params_setting_;
  
  //num of levels in bp processing hierarchy
  const unsigned int num_levels_;

  //stores the current parallel parameters for each processing kernel
  std::array<std::vector<std::array<unsigned int, 2>>, beliefprop::kNumKernels> parallel_dims_each_kernel_;
  
  //mapping of parallel parameters to runtime for each kernel at each level and total runtime
  std::array<std::vector<std::map<std::array<unsigned int, 2>, double>>, (beliefprop::kNumKernels + 1)> p_params_to_run_time_each_kernel_;
};

#endif //BP_PARALLEL_PARAMS_H
