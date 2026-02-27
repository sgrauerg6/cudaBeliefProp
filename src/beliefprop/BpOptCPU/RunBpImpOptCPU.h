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
 * @file RunBpImpOptCPU.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of RunBpImp to run optimized CPU
 * implementation of belief propagation on a given stereo set as defined
 * by reference and test image file paths
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_BP_IMP_OPT_CPU_H_
#define RUN_BP_IMP_OPT_CPU_H_

#include <string>
#include <memory>
#include <array>
#include "BpRunProcessing/RunBpImp.h"
#include "BpRunProcessing/ProcessBpDevice.h"
#include "BpRunProcessing/ParallelParamsBp.h"
#include "RunEval/RunTypeConstraints.h"
#include "SmoothImageOptCPU.h"
#include "ProcessBpOptCPU.h"

/**
 * @brief Child class of RunBpImp to run optimized CPU implementation of belief propagation on a
 * given stereo set as defined by reference and test image file paths
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpImpOptCPU final : public RunBpImp<T, DISP_VALS, ACCELERATION> {
public:
  std::string RunDescription() const override { return std::string(run_cpu::kOptCPUDesc); }

  //run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
  std::optional<beliefprop::BpRunOutput> operator()(
    const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings,
    const ParallelParams& parallel_params) const override;
};

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline std::optional<beliefprop::BpRunOutput> RunBpImpOptCPU<T, DISP_VALS, ACCELERATION>::operator()(
  const std::array<std::string, 2>& ref_test_image_path,
  const beliefprop::BpSettings& alg_settings,
  const ParallelParams& parallel_params) const
{
  //set number of threads to use when running code in parallel using OpenMP from input parallel parameters
  //current setting on CPU is to execute all parallel processing in a run using the same number of parallel threads
  const unsigned int nthreads = 
    parallel_params.OptParamsForKernel({static_cast<size_t>(beliefprop::BpKernel::kBlurImages), 0})[0];
  #ifndef __APPLE__
    omp_set_num_threads(nthreads);
  #endif //__APPLE__

  //add settings for current run to output data
  RunData run_data;
  run_data.AddDataWHeader(
    std::string(run_cpu::kNumParallelCPUThreadsHeader),
    nthreads);
  run_data.AddDataWHeader(
    std::string(run_cpu::kCPUVectorizationHeader),
    std::string(run_environment::AccelerationString<ACCELERATION>()));

  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  auto process_set_output = this->ProcessImplementation(
    ref_test_image_path,
    alg_settings, 
    beliefprop::BpOnDevice<T, DISP_VALS, ACCELERATION>{
      std::make_unique<SmoothImageOptCPU>(parallel_params),
      std::make_unique<ProcessBpOptCPU<T, DISP_VALS, ACCELERATION>>(parallel_params),
      std::make_unique<MemoryManagement<T>>(),
      std::make_unique<MemoryManagement<float>>()});
  if (process_set_output) {
    run_data.AppendData(std::move(process_set_output->run_data));
    process_set_output->run_data = std::move(run_data);
  }

  return process_set_output;
}

#endif /* RUN_BP_IMP_OPT_CPU_H_ */
