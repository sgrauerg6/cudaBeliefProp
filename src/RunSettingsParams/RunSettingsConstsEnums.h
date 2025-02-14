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
 * @file RunSettingsConstsEnums.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with constants and enums related to run
 * environment and settings for run
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_SETTINGS_CONSTS_ENUMS_H_
#define RUN_SETTINGS_CONSTS_ENUMS_H_

#include <string_view>
#include <map>

#if defined(OPTIMIZED_CUDA_RUN)
#include "RunImpCUDA/RunCUDASettings.h"
#endif //OPTIMIZED_CUDA_RUN

//set alias for data type used for half-precision
#if defined(OPTIMIZED_CPU_RUN)
#if defined(COMPILING_FOR_ARM)
#include <arm_neon.h> //needed for float16_t type
using halftype = float16_t;
#else
#include "RunImpCPU/RunCPUSettings.h"
using halftype = short;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CPU_RUN

/**
 * @brief Constants and enums related to run environment
 * and settings for run
 */
namespace run_environment {

/** @brief Constant string for acceleration */
constexpr std::string_view kAccelerationDescHeader{"Acceleration"};

/** @brief Enum for acceleration setting */
enum class AccSetting {
  kAVX512_F16, kAVX512, kAVX256_F16, kAVX256, kNEON, kCUDA, kNone
};

/**
 * @brief Get string corresponding to acceleration method at compile time
 * 
 * @tparam ACCELERATION_SETTING 
 * @return String view corresponding to acceleration method
 */
template <AccSetting ACCELERATION_SETTING>
constexpr std::string_view AccelerationString() {
  if constexpr (ACCELERATION_SETTING == AccSetting::kNEON) { return "NEON"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX256) { return "AVX256"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX256_F16) { return "AVX256_float16Vect"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX512) { return "AVX512"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX512_F16) { return "AVX512_float16Vect"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kCUDA) { return "CUDA"; }
  else { return "DEFAULT"; }
}

/**
 * @brief Get string corresponding to acceleration method at run time
 * 
 * @param acceleration_setting 
 * @return String view corresponding to acceleration method
 */
inline std::string_view AccelerationString(AccSetting acceleration_setting) {
  if (acceleration_setting == AccSetting::kNEON) { return AccelerationString<AccSetting::kNEON>(); }
  else if (acceleration_setting == AccSetting::kAVX256) { return AccelerationString<AccSetting::kAVX256>(); }
  else if (acceleration_setting == AccSetting::kAVX256_F16) { return AccelerationString<AccSetting::kAVX256_F16>(); }
  else if (acceleration_setting == AccSetting::kAVX512) { return AccelerationString<AccSetting::kAVX512>(); }
  else if (acceleration_setting == AccSetting::kAVX512_F16) { return AccelerationString<AccSetting::kAVX512_F16>(); }
  else if (acceleration_setting == AccSetting::kCUDA) { return AccelerationString<AccSetting::kCUDA>(); }
  else { return "DEFAULT"; }
}

/** @brief Mapping from data size to data type string */
const std::map<std::size_t, std::string_view> kDataSizeToNameMap{
  {sizeof(float), "FLOAT"},
  {sizeof(double), "DOUBLE"},
  {sizeof(short), "HALF"}
};

/** @brief Enum that specifies whether or not to use templated counts for the
 *  number of iterations in processing loops or to run implementation with and
 *  without templated iteration counts
 *  Templated counts for number of loop iterations can help with optimization
 *  but requires that the number of iterations be known at compile time */
enum class TemplatedItersSetting {
  kRunOnlyTempated,
  kRunOnlyNonTemplated,
  kRunTemplatedAndNotTemplated
};

/** @brief Enum for parallel parameters settings in run */
enum class ParallelParamsSetting { kDefault, kOptimized };

/** @brief Enum to specify if optimizing parallel parameters per kernel or
 *  using same parallel parameters across all kernels in run
 *  In initial testing optimizing per kernel is faster on GPU and using same
 *  parallel parameters across all kernels is faster on CPU */
enum class OptParallelParamsSetting {
  kSameParallelParamsAllKernels,
  kAllowDiffKernelParallelParamsInRun
};

//constants for header and descriptions corresponding to optimized parallel
//parameters setting
constexpr std::string_view kPParamsPerKernelSettingHeader{"Parallel Params Per Kernel Setting"};
constexpr std::string_view kPParamsSameEveryKernelDesc{"Same Across Kernels"};
constexpr std::string_view kPParamsSetEachKernelDesc{"Set Per Kernel"};
const std::map<OptParallelParamsSetting, std::string_view> kOptPParmsSettingToDesc {
  {OptParallelParamsSetting::kSameParallelParamsAllKernels, kPParamsSameEveryKernelDesc},
  {OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun, kPParamsSetEachKernelDesc}
};

//constants for headers corresponding to run settings
constexpr std::string_view kCPUThreadsPinnedHeader{"CPU Threads Pinned To Socket"};
constexpr std::string_view kOmpPlacesHeader{"OMP_PLACES"};
constexpr std::string_view kOmpProcBindHeader{"OMP_PROC_BIND"};
constexpr std::string_view kNumCPUThreadsHeader{"Total number of CPU threads"};
constexpr std::string_view kBytesAlignMemHeader{"BYTES_ALIGN_MEMORY"};

};

#endif //RUN_SETTINGS_CONSTS_ENUMS_H_