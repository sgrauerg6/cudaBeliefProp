/*
 * RunBpOnStereoSetOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUN_BP_ON_STEREO_SET_OPTIMIZED_CPU_H_
#define RUN_BP_ON_STEREO_SET_OPTIMIZED_CPU_H_

#include <string>
#include <memory>
#include <array>
#include "BpRunProcessing/RunBpOnStereoSet.h"
#include "BpRunProcessing/ProcessBp.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunEval/RunTypeConstraints.h"
#include "SmoothImageCPU.h"
#include "ProcessBpOptimizedCPU.h"

namespace beliefprop {
  constexpr std::string_view kBpOptimizeCPUDesc{"Optimized CPU"};
  constexpr std::string_view kNumParallelCPUThreadsHeader{"Number of parallel CPU threads in run"};
  constexpr std::string_view kCPUVectorizationHeader{"Vectorization"};
};

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpOnStereoSetOptimizedCPU final : public RunBpOnStereoSet<T, DISP_VALS, ACCELERATION> {
public:
  std::string BpRunDescription() const override { return std::string(beliefprop::kBpOptimizeCPUDesc); }

  //run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings,
    const ParallelParams& parallel_params) const override;
};

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline std::optional<ProcessStereoSetOutput> RunBpOnStereoSetOptimizedCPU<T, DISP_VALS, ACCELERATION>::operator()(
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
  run_data.AddDataWHeader(std::string(beliefprop::kNumParallelCPUThreadsHeader), nthreads);
  run_data.AddDataWHeader(std::string(beliefprop::kCPUVectorizationHeader),
    std::string(run_environment::AccelerationString<ACCELERATION>()));

  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  auto process_set_output = this->ProcessStereoSet(ref_test_image_path, alg_settings, 
    BpOnDevice<T, DISP_VALS, ACCELERATION>{
      std::make_unique<SmoothImageCPU>(parallel_params),
      std::make_unique<ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>>(parallel_params),
      std::make_unique<MemoryManagement<T>>(),
      std::make_unique<MemoryManagement<float>>()});
  if (process_set_output) {
    run_data.AppendData(process_set_output->run_data);
    process_set_output->run_data = run_data;
  }

  return process_set_output;
}

#endif /* RUN_BP_ON_STEREO_SET_OPTIMIZED_CPU_H_ */
