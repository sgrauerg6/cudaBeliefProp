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
 * @file RunCUDASettings.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with constants and functions to get CUDA device
 * properties as well as default and test parallel parameters to use in CUDA
 * implementation run optimization
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_CUDA_SETTINGS_H_
#define RUN_CUDA_SETTINGS_H_

#include <vector>
#include <array>
#include "RunEval/RunData.h"

//set data type used for half-precision with CUDA
#if defined(USE_BFLOAT16_FOR_HALF_PRECISION)
#include <cuda_bf16.h>
using halftype = __nv_bfloat16;
#else
#include <cuda_fp16.h>
using halftype = half;
#endif //USE_BFLOAT16_FOR_HALF_PRECISION

/**
 * @brief Namespace with constants and functions to get CUDA device properties as
 * well as default and test parallel parameters to use in CUDA implementation run
 * optimization
 */
namespace run_cuda {

constexpr std::string_view kOptimizeCUDADesc{"CUDA"};
constexpr std::string_view kCUDAVersionHeader{"Cuda version"};
constexpr std::string_view kCUDARuntimeHeader{"Cuda Runtime Version"};

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
  run_data.AddDataWHeader(std::string(kCUDAVersionHeader),
    std::to_string(cuda_version_driver_runtime[0]));
  run_data.AddDataWHeader(std::string(kCUDARuntimeHeader),
    std::to_string(cuda_version_driver_runtime[1]));
  return run_data;
}

/** @brief Parallel parameter options to run to retrieve optimized parallel parameters in CUDA implementation
 *  Parallel parameter corresponds to thread block dimensions in CUDA implementation */
#if defined(LIMITED_TEST_PARAMS)
const std::vector<std::array<unsigned int, 2>> kParallelParameterOptions{
  {16, 1}, {32, 4}, {64, 4}};
#else
const std::vector<std::array<unsigned int, 2>> kParallelParameterOptions{
  {16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5},
  {32, 6}, {32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4},
  {128, 1}, {128, 2}, {256, 1}, {32, 10}, {32, 12}, {32, 14}, {32, 16},
  {64, 5}, {64, 6}, {64, 7}, {64, 8}, {128, 3}, {128, 4}, {256, 2}};
#endif //LIMITED_TEST_PARAMS

/** @brief Default thread block dimensions (which is what parallel parameters
 *  corresponds to in CUDA implementation) */
constexpr std::array<unsigned int, 2> kParallelParamsDefault{{32, 4}};

};

#endif /* RUN_CUDA_SETTINGS_H_ */
