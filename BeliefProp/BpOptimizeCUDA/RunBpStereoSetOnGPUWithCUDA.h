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

#ifndef RUN_BP_STEREO_STEREO_SET_ON_GPU_WITH_CUDA_H
#define RUN_BP_STEREO_STEREO_SET_ON_GPU_WITH_CUDA_H

#include <array>
#include <cuda_runtime.h>
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunImpCUDA/RunImpCUDAMemoryManagement.h"
#include "ProcessCUDABP.h"
#include "SmoothImageCUDA.h"

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
      std::string(prop.name) + " with " + std::to_string(prop.multiProcessorCount) + " multiprocessors");
    run_data.AddDataWHeader("Cuda version", std::to_string(cuda_version_driver_runtime[0]));
    run_data.AddDataWHeader("Cuda Runtime Version", std::to_string(cuda_version_driver_runtime[1]));
    return run_data;
  }
};

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpStereoSetOnGPUWithCUDA final : public RunBpStereoSet<T, DISP_VALS, ACCELERATION>
{
public:
  std::string BpRunDescription() const override { return "CUDA"; }

  //run the disparity map estimation BP on a set of stereo images and save the results between each set of images
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings, 
    const ParallelParams& parallel_params) override
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
        std::make_unique<ProcessCUDABP<T, DISP_VALS, ACCELERATION>>(parallel_params),
        std::make_unique<RunImpCUDAMemoryManagement<T>>(),
        std::make_unique<RunImpCUDAMemoryManagement<float>>()});
    if (process_set_output) {
      run_data.AppendData(process_set_output->run_data);
      process_set_output->run_data = run_data;
    }
    
    return process_set_output;
  }
};

#endif //RUN_BP_STEREO_IMAGE_SERIES_HEADER_CUH
