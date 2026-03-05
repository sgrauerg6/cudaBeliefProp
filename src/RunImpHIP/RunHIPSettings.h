/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file RunHIPSettings.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with constants and functions to get HIP device
 * properties as well as default and test parallel parameters to use in HIP
 * implementation run optimization
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_HIP_SETTINGS_H_
#define RUN_HIP_SETTINGS_H_

#include <vector>
#include <array>
#include <set>
#include "RunEval/RunData.h"

//set data type used for half-precision with CUDA
#if defined(USE_BFLOAT16_FOR_HALF_PRECISION)
#include <hip_bf16.h>
using halftype = __hip_bfloat16;
#else
#include <hip_fp16.h>
using halftype = half;
#endif //USE_BFLOAT16_FOR_HALF_PRECISION

/**
 * @brief Namespace with constants and functions to get HIP device properties
 * as well as default and test parallel parameters to use in HIP
 * implementation run optimization
 */
namespace run_hip {

constexpr std::string_view kHIPDesc{"HIP"};
constexpr std::string_view kHIPVersionHeader{"HIP version"};
constexpr std::string_view kHIPRuntimeHeader{"HIP Runtime Version"};

inline RunData retrieveDeviceProperties(int num_device)
{
  hipDeviceProp prop;
  std::array<int, 2> hip_version_driver_runtime;
  hipGetDeviceProperties(&prop, num_device);
  hipDriverGetVersion(&(hip_version_driver_runtime[0]));
  hipRuntimeGetVersion(&(hip_version_driver_runtime[1]));

  RunData run_data;
  run_data.AddDataWHeader(
    "Device " + std::to_string(num_device),
    std::string(prop.name) + " with " + 
      std::to_string(prop.multiProcessorCount) +
      " multiprocessors");
  run_data.AddDataWHeader(
    std::string(kHIPVersionHeader),
    std::to_string(hip_version_driver_runtime[0]));
  run_data.AddDataWHeader(
    std::string(kHIPRuntimeHeader),
    std::to_string(hip_version_driver_runtime[1]));
  return run_data;
}

/** @brief Default thread block dimensions (which is what parallel parameters
 *  corresponds to in HIP implementation) */
constexpr std::array<unsigned int, 2> kParallelParamsDefault{{32, 4}};

#if defined(DEFAULT_PARALLEL_PARAMS_ONLY)

/** @brief Empty parallel parameter alternative options since setting is to
  *  only use default parallel parameters. */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{};

#elif defined(LIMITED_ALT_PARALLEL_PARAMS)

/** @brief Parallel parameter alternative options to run to retrieve optimized parallel
 *  parameters in HIP implementation
 *  Parallel parameter corresponds to thread block dimensions in HIP implementation
 *  OK to include default parallel parameters in alternative options but not required */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{
  {16, 1}, {32, 4}, {64, 4}};

#else

/** @brief Parallel parameter alternative options to run to retrieve optimized parallel
 *  parameters in HIP implementation
 *  Parallel parameter corresponds to thread block dimensions in HIP implementation
 *  OK to include default parallel parameters in alternative options but not required */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{
  {16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5},
  {32, 6}, {32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4},
  {128, 1}, {128, 2}, {256, 1}, {32, 10}, {32, 12}, {32, 14}, {32, 16},
  {64, 5}, {64, 6}, {64, 7}, {64, 8}, {128, 3}, {128, 4}, {256, 2}};

#endif

};

#endif /* RUN_HIP_SETTINGS_H_ */
