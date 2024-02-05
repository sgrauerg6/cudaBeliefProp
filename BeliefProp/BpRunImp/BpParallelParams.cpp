/*
 * BpParallelParams.cpp
 *
 *  Created on: Feb 4, 2024
 *      Author: scott
 */

#include "BpParallelParams.h"

//constructor to set parallel parameters with default dimensions for each kernel
BpParallelParams::BpParallelParams(run_environment::OptParallelParamsSetting optParallelParamsSetting,
    unsigned int numLevels, const std::array<unsigned int, 2>& defaultPDims) : optParallelParamsSetting_{optParallelParamsSetting}, numLevels_{numLevels}
{
  setParallelDims(defaultPDims);
  //set up mapping of parallel parameters to runtime for each kernel at each level and total runtime
  for (unsigned int i=0; i < beliefprop::NUM_KERNELS; i++) {
    //set to vector length for each kernel to corresponding vector length of kernel in parallelParams.parallelDimsEachKernel_
    pParamsToRunTimeEachKernel_[i] = std::vector<std::map<std::array<unsigned int, 2>, double>>(parallelDimsEachKernel_[i].size()); 
  }
  pParamsToRunTimeEachKernel_[beliefprop::NUM_KERNELS] = std::vector<std::map<std::array<unsigned int, 2>, double>>(1); 
}

//set parallel parameters for each kernel to the same input dimensions
void BpParallelParams::setParallelDims(const std::array<unsigned int, 2>& tbDims) {
  parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES] = {tbDims};
  parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL] = std::vector<std::array<unsigned int, 2>>(numLevels_, tbDims);
  parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS] = {tbDims};
  parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL] = std::vector<std::array<unsigned int, 2>>(numLevels_, tbDims);
  parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL] = std::vector<std::array<unsigned int, 2>>(numLevels_, tbDims);
  parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP] = {tbDims};
}

//add current parallel parameters to data for current run
RunData BpParallelParams::runData() const {
  RunData currRunData;
  //show parallel parameters for each kernel
  currRunData.addDataWHeader("Blur Images Parallel Dimensions",
    std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][0]) + " x " +
    std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][1]));
  currRunData.addDataWHeader("Init Message Values Parallel Dimensions",
    std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][0]) + " x " +
    std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][1]));
  for (unsigned int level=0; level < parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL].size(); level++) {
    currRunData.addDataWHeader("Level " + std::to_string(level) + " Data Costs Parallel Dimensions",
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][0]) + " x " +
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][1]));
  }
  for (unsigned int level=0; level < parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL].size(); level++) {
    currRunData.addDataWHeader("Level " + std::to_string(level) + " BP Thread Parallel Dimensions",
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][level][0]) + " x " +
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][level][1]));
  }
  for (unsigned int level=0; level < parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL].size(); level++) {
    currRunData.addDataWHeader("Level " + std::to_string(level) + " Copy Thread Parallel Dimensions",
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][level][0]) + " x " +
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][level][1]));
  }
  currRunData.addDataWHeader("Get Output Disparity Parallel Dimensions",
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][0]) + " x " +
      std::to_string(parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][1]));

  return currRunData;
}

//add results from run with same specified parallel parameters used every parallel component
void BpParallelParams::addTestResultsForParallelParams(const std::array<unsigned int, 2>& pParamsCurrRun, const RunData& currRunData)
{
  const std::string NUM_RUNS_IN_PARENS{"(" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"};
  if (optParallelParamsSetting_ == run_environment::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
    for (unsigned int level=0; level < numLevels_; level++) {
      pParamsToRunTimeEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][pParamsCurrRun] =
        std::stod(currRunData.getData(beliefprop::LEVEL_DCOST_BPTIME_CTIME_NAMES[level][0] + " " + NUM_RUNS_IN_PARENS));
      pParamsToRunTimeEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][level][pParamsCurrRun] = 
        std::stod(currRunData.getData(beliefprop::LEVEL_DCOST_BPTIME_CTIME_NAMES[level][1] + " " + NUM_RUNS_IN_PARENS));
      pParamsToRunTimeEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][level][pParamsCurrRun] =
        std::stod(currRunData.getData(beliefprop::LEVEL_DCOST_BPTIME_CTIME_NAMES[level][2] + " " + NUM_RUNS_IN_PARENS));
    }
    pParamsToRunTimeEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][pParamsCurrRun] =
      std::stod(currRunData.getData(beliefprop::timingNames.at(beliefprop::Runtime_Type::SMOOTHING) + " " + NUM_RUNS_IN_PARENS));
    pParamsToRunTimeEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][pParamsCurrRun] =
      std::stod(currRunData.getData(beliefprop::timingNames.at(beliefprop::Runtime_Type::INIT_MESSAGES_KERNEL) + " " + NUM_RUNS_IN_PARENS));
    pParamsToRunTimeEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][pParamsCurrRun] =
      std::stod(currRunData.getData(beliefprop::timingNames.at(beliefprop::Runtime_Type::OUTPUT_DISPARITY) + " " + NUM_RUNS_IN_PARENS));
  }
  //get total runtime
  pParamsToRunTimeEachKernel_[beliefprop::NUM_KERNELS][0][pParamsCurrRun] =
    std::stod(currRunData.getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
}

//retrieve optimized parameters from results across multiple runs with different parallel parameters and set current parameters
//to retrieved optimized parameters
void BpParallelParams::setOptimizedParams() {
  if (optParallelParamsSetting_ == run_environment::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
    for (unsigned int numKernelSet = 0; numKernelSet < parallelDimsEachKernel_.size(); numKernelSet++) {
      //retrieve and set optimized parallel parameters for final run
      //std::min_element used to retrieve parallel parameters corresponding to lowest runtime from previous runs
      std::transform(pParamsToRunTimeEachKernel_[numKernelSet].begin(),
                     pParamsToRunTimeEachKernel_[numKernelSet].end(), 
                     parallelDimsEachKernel_[numKernelSet].begin(),
                     [](const auto& tDimsToRunTimeCurrLevel) /*-> std::array<unsigned int, 2>*/ { 
                       return (std::min_element(tDimsToRunTimeCurrLevel.begin(), tDimsToRunTimeCurrLevel.end(),
                                                [](const auto& a, const auto& b) { return a.second < b.second; }))->first; });
    }
  }
  else {
    //set optimized parallel parameters for all kernels to parallel parameters that got the best runtime across all kernels
    //seems like setting different parallel parameters for different kernels on GPU decrease runtime but increases runtime on CPU
    const auto bestParallelParams = std::min_element(pParamsToRunTimeEachKernel_[beliefprop::NUM_KERNELS][0].begin(),
                                                     pParamsToRunTimeEachKernel_[beliefprop::NUM_KERNELS][0].end(),
                                                     [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    setParallelDims(bestParallelParams);
  }
}
