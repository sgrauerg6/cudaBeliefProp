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
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "SmoothImageCPU.h"
#include "ProcessOptimizedCPUBP.h"

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
class RunBpStereoOptimizedCPU final : public RunBpStereoSet<T, DISP_VALS, VECTORIZATION> {
public:
  std::string BpRunDescription() const override { return "Optimized CPU"; }

  //run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings,
    const ParallelParams& parallel_params) const override;
};

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline std::optional<ProcessStereoSetOutput> RunBpStereoOptimizedCPU<T, DISP_VALS, VECTORIZATION>::operator()(
  const std::array<std::string, 2>& ref_test_image_path,
  const beliefprop::BpSettings& alg_settings, const ParallelParams& parallel_params) const
{
  //set number of threads to use when running code in parallel using OpenMP from input parallel parameters
  //current setting on CPU is to execute all parallel processing in a run using the same number of parallel threads
  const unsigned int nthreads = 
    parallel_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0];
  omp_set_num_threads(nthreads);

  //add settings for current run to output data
  RunData run_data;
  run_data.AddDataWHeader("Number of parallel CPU threads in run", nthreads);
  run_data.AddDataWHeader("Vectorization", std::string(run_environment::AccelerationString<VECTORIZATION>()));

  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  auto process_set_output = this->ProcessStereoSet(ref_test_image_path, alg_settings, 
    BpOnDevice<T, DISP_VALS, VECTORIZATION>{
      std::make_unique<SmoothImageCPU>(parallel_params),
      std::make_unique<ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>>(parallel_params),
      std::make_unique<RunImpMemoryManagement<T>>(),
      std::make_unique<RunImpMemoryManagement<float>>()});
  if (process_set_output) {
    run_data.AppendData(process_set_output->run_data);
    process_set_output->run_data = run_data;
  }

  return process_set_output;
}

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
