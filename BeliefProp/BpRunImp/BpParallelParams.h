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
#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunImp/ParallelParams.h"

//structure containing parameters including parallelization parameters
//to use at each BP level
class BpParallelParams : public ParallelParams {
public:
  //constructor to set parallel parameters with default dimensions for each kernel
  BpParallelParams(run_environment::OptParallelParamsSetting optParallelParamsSetting,
    unsigned int numLevels, const std::array<unsigned int, 2>& defaultPDims);

  //set parallel parameters for each kernel to the same input dimensions
  void setParallelDims(const std::array<unsigned int, 2>& tbDims) override;

  //add current parallel parameters to data for current run
  RunData runData() const override;

  //add results from run with same specified parallel parameters used every parallel component
  void addTestResultsForParallelParams(const std::array<unsigned int, 2>& pParamsCurrRun, const RunData& currRunData);

  //retrieve optimized parameters from results across multiple runs with different parallel parameters and set current parameters
  //to retrieved optimized parameters
  void setOptimizedParams() override;

  //get optimized parallel parameters for parallel processing kernel for kernel that is indexed as an array of two unsigned integers
  std::array<unsigned int, 2> getOptParamsForKernel(const std::array<unsigned int, 2>& kernelLocation) const override {
    return parallelDimsEachKernel_[kernelLocation[0]][kernelLocation[1]];
  }

  //stores the current parallel parameters for each processing kernel
  std::array<std::vector<std::array<unsigned int, 2>>, beliefprop::NUM_KERNELS> parallelDimsEachKernel_;

private:
  const run_environment::OptParallelParamsSetting optParallelParamsSetting_;
  const unsigned int numLevels_;
  //mapping of parallel parameters to runtime for each kernel at each level and total runtime
  std::array<std::vector<std::map<std::array<unsigned int, 2>, double>>, (beliefprop::NUM_KERNELS + 1)> pParamsToRunTimeEachKernel_;
};

#endif //BP_PARALLEL_PARAMS_H
