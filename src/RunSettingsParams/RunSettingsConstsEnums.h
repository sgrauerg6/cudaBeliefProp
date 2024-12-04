/*
 * RunSettingsConstsEnums.h
 *
 *  Created on: Dec 2, 2024
 *      Author: scott
 */

#ifndef RUN_SETTINGS_CONSTS_ENUMS_H_
#define RUN_SETTINGS_CONSTS_ENUMS_H_

#include <string_view>
#include <map>

#ifdef OPTIMIZED_CUDA_RUN
#include "RunImpCUDA/RunCUDASettings.h"
#endif //OPTIMIZED_CUDA_RUN

//set alias for data type used for half-precision
#ifdef OPTIMIZED_CPU_RUN
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
using halftype = float16_t;
#else
using halftype = short;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CPU_RUN

//constants and enums related to run environment
//and settings for run
namespace run_environment {

//enum for acceleration setting
enum class AccSetting {
  kNone, kAVX256, kAVX512, kNEON, kCUDA
};

//get string corresponding to acceleration method at compile time
template <AccSetting ACCELERATION_SETTING>
constexpr std::string_view AccelerationString() {
  if constexpr (ACCELERATION_SETTING == AccSetting::kNEON) { return "NEON"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX256) { return "AVX256"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX512) { return "AVX512"; }
  else { return "DEFAULT"; }
}

//get string corresponding to acceleration method at run time
inline std::string_view AccelerationString(AccSetting acceleration_setting) {
  if (acceleration_setting == AccSetting::kNEON) { return AccelerationString<AccSetting::kNEON>(); }
  else if (acceleration_setting == AccSetting::kAVX256) { return AccelerationString<AccSetting::kAVX256>(); }
  else if (acceleration_setting == AccSetting::kAVX512) { return AccelerationString<AccSetting::kAVX512>(); }
  else { return "DEFAULT"; }
}

//mapping from data size to data type string
const std::map<std::size_t, std::string_view> kDataSizeToNameMap{
  {sizeof(float), "FLOAT"},
  {sizeof(double), "DOUBLE"},
  {sizeof(short), "HALF"}
};

//enum that specifies whether or not to use templated counts for the number of iterations in processing
//loops or to run implementation with and without templated iteration counts
//templated counts for number of loop iterations can help with optimization but requires that the number of
//iterations be known at compile time
enum class TemplatedItersSetting {
  kRunOnlyTempated,
  kRunOnlyNonTemplated,
  kRunTemplatedAndNotTemplated
};

//enum for parallel parameters settings in run
enum class ParallelParamsSetting { kDefault, kOptimized };

//enum to specify if optimizing parallel parameters per kernel or using same parallel parameters across all kernels in run
//in initial testing optimizing per kernel is faster on GPU and using same parallel parameters across all kernels is faster
//on CPU
enum class OptParallelParamsSetting {
  kSameParallelParamsAllKernels,
  kAllowDiffKernelParallelParamsInRun
};

//constants for headers corresponding to run settings
constexpr std::string_view kCPUThreadsPinnedHeader{"CPU Threads Pinned To Socket"};
constexpr std::string_view kOmpPlacesHeader{"OMP_PLACES"};
constexpr std::string_view kOmpProcBindHeader{"OMP_PROC_BIND"};
constexpr std::string_view kNumCPUThreadsHeader{"Total number of CPU threads"};
constexpr std::string_view kBytesAlignMemHeader{"BYTES_ALIGN_MEMORY"};
constexpr std::string_view kNumDataAlignWidthHeader{"NUM_DATA_ALIGN_WIDTH"};

};

#endif //RUN_SETTINGS_CONSTS_ENUMS_H_