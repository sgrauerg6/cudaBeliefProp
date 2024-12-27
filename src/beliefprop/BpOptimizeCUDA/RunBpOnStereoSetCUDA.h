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
 * @file RunBpOnStereoSetCUDA.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of RunBpOnStereoSet to run CUDA implementation
 * of belief propagation on a given stereo set as defined by reference and test
 * image file paths
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_BP_ON_STEREO_STEREO_CUDA_H
#define RUN_BP_ON_STEREO_STEREO_CUDA_H

#include <array>
#include <cuda_runtime.h>
#include "BpRunProcessing/RunBpOnStereoSet.h"
#include "BpRunProcessing/ProcessBp.h"
#include "BpRunProcessing/ParallelParamsBp.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImpCUDA/MemoryManagementCUDA.h"
#include "ProcessBpCUDA.h"
#include "SmoothImageCUDA.h"

/**
 * @brief Child class of RunBpOnStereoSet to run CUDA implementation of belief
 * propagation on a given stereo set as defined by reference and test image
 * file paths
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpOnStereoSetCUDA final : public RunBpOnStereoSet<T, DISP_VALS, ACCELERATION>
{
public:
  std::string BpRunDescription() const override { 
    return std::string(run_cuda::kOptimizeCUDADesc); }

  /**
   * @brief Run the disparity map estimation BP on a set of stereo images
   * and save the results between each set of images
   * 
   * @param ref_test_image_path 
   * @param alg_settings 
   * @param parallel_params 
   * @return std::optional<beliefprop::BpRunOutput> 
   */
  std::optional<beliefprop::BpRunOutput> operator()(
    const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings, 
    const ParallelParams& parallel_params) const override
  {
    //return no value if acceleration setting is not CUDA
    if constexpr (ACCELERATION != run_environment::AccSetting::kCUDA) {
      return {};
    }

    //generate struct with pointers to objects for running CUDA implementation and call
    //function to run CUDA implementation
    RunData run_data;
    run_data.AppendData(run_cuda::retrieveDeviceProperties(0));
    auto process_set_output = 
      this->ProcessStereoSet(
        ref_test_image_path,
        alg_settings,
        beliefprop::BpOnDevice<T, DISP_VALS, ACCELERATION>{
          std::make_unique<SmoothImageCUDA>(parallel_params),
          std::make_unique<ProcessBpCUDA<T, DISP_VALS, ACCELERATION>>(parallel_params),
          std::make_unique<MemoryManagementCUDA<T>>(),
          std::make_unique<MemoryManagementCUDA<float>>()});
    if (process_set_output) {
      run_data.AppendData(process_set_output->run_data);
      process_set_output->run_data = run_data;
    }
    
    return process_set_output;
  }
};

#endif //RUN_BP_ON_STEREO_STEREO_CUDA_H
