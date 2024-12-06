/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//Declares the methods to run Stereo BP on a series of images

#ifndef RUN_BP_ON_STEREO_STEREO_CUDA_H
#define RUN_BP_ON_STEREO_STEREO_CUDA_H

#include <array>
#include <cuda_runtime.h>
#include "BpRunProcessing/RunBpOnStereoSet.h"
#include "BpRunProcessing/ProcessBp.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImpCUDA/RunImpCUDAMemoryManagement.h"
#include "ProcessBpCUDA.h"
#include "SmoothImageCUDA.h"

namespace beliefprop {
  constexpr std::string_view kBpOptimizeCUDADesc{"CUDA"};
  constexpr std::string_view kCUDAVersionHeader{"Cuda version"};
  constexpr std::string_view kCUDARuntimeHeader{"Cuda Runtime Version"};
};

namespace bp_cuda_device
{
  inline RunData retrieveDeviceProperties(int num_device)
  {
    cudaDeviceProp prop;
    std::array<int, 2> cuda_version_driver_runtime;
    cudaGetDeviceProperties(&prop, num_device);
    cudaDriverGetVersion(&(cuda_version_driver_runtime[0]));
    cudaRuntimeGetVersion(&(cuda_version_driver_runtime[1]));

    RunData run_data;
    run_data.AddDataWHeader("Device " + std::to_string(num_device),
      std::string(prop.name) + " with " + std::to_string(prop.multiProcessorCount) +
      " multiprocessors");
    run_data.AddDataWHeader(std::string(beliefprop::kCUDAVersionHeader),
      std::to_string(cuda_version_driver_runtime[0]));
    run_data.AddDataWHeader(std::string(beliefprop::kCUDARuntimeHeader),
      std::to_string(cuda_version_driver_runtime[1]));
    return run_data;
  }
};

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpOnStereoSetCUDA final : public RunBpOnStereoSet<T, DISP_VALS, ACCELERATION>
{
public:
  std::string BpRunDescription() const override { 
    return std::string(beliefprop::kBpOptimizeCUDADesc); }

  //run the disparity map estimation BP on a set of stereo images and save the results between each set of images
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& ref_test_image_path,
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
    run_data.AppendData(bp_cuda_device::retrieveDeviceProperties(0));
    auto process_set_output = this->ProcessStereoSet(ref_test_image_path, alg_settings,
      BpOnDevice<T, DISP_VALS, ACCELERATION>{
        std::make_unique<SmoothImageCUDA>(parallel_params),
        std::make_unique<ProcessBpCUDA<T, DISP_VALS, ACCELERATION>>(parallel_params),
        std::make_unique<RunImpCUDAMemoryManagement<T>>(),
        std::make_unique<RunImpCUDAMemoryManagement<float>>()});
    if (process_set_output) {
      run_data.AppendData(process_set_output->run_data);
      process_set_output->run_data = run_data;
    }
    
    return process_set_output;
  }
};

#endif //RUN_BP_ON_STEREO_STEREO_CUDA_H
