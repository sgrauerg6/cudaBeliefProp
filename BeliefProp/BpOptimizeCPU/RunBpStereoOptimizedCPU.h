/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include <string>
#include <memory>
#include <array>
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "SmoothImageCPU.h"
#include "ProcessOptimizedCPUBP.h"

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
class RunBpStereoOptimizedCPU : public RunBpStereoSet<T, DISP_VALS, VECTORIZATION> {
public:
  std::string getBpRunDescription() override { return "Optimized CPU"; }

  //run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings,
    const ParallelParams& parallelParams) override;
};

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline std::optional<ProcessStereoSetOutput> RunBpStereoOptimizedCPU<T, DISP_VALS, VECTORIZATION>::operator()(const std::array<std::string, 2>& refTestImagePath,
  const beliefprop::BPsettings& algSettings, const ParallelParams& parallelParams)
{
  //set number of threads to use when running code in parallel using OpenMP from input parallel parameters
  //current setting on CPU is to execute all parallel processing in a run using the same number of parallel threads
  unsigned int nthreads = parallelParams.getOptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::BLUR_IMAGES), 0})[0];
  omp_set_num_threads(nthreads);

  //add settings for current run to output data
  RunData runData;
  runData.addDataWHeader("Number of parallel CPU threads in run", nthreads);
  runData.addDataWHeader("Vectorization", std::string(run_environment::accelerationString<VECTORIZATION>()));

  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  auto procSetOutput = this->processStereoSet(refTestImagePath, algSettings, 
    BpOnDevice<T, DISP_VALS, VECTORIZATION>{
      std::make_unique<SmoothImageCPU>(parallelParams),
      std::make_unique<ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>>(parallelParams),
      std::make_unique<RunImpMemoryManagement<T>>(),
      std::make_unique<RunImpMemoryManagement<float>>()});
  if (procSetOutput) {
    runData.appendData(procSetOutput->runData);
    procSetOutput->runData = runData;
  }

  return procSetOutput;
}

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
