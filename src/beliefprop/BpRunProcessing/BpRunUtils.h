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
 * @file BpRunUtils.h
 * @author Scott Grauer-Gray
 * @brief File with namespace for enums, constants, structures, and
 * functions specific to belief propagation processing
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BP_RUN_UTILS_H
#define BP_RUN_UTILS_H

#include <string>
#include <string_view>
#include <array>
#include <limits>
#include "RunSettingsParams/RunSettingsConstsEnums.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunData.h"

/**
 * @brief Namespace for enums, constants, structures, and
 * functions specific to belief propagation processing
 */
namespace beliefprop {

/** @brief High value for type to use if initializing to "high" value */
template <typename T>
const T kHighValBp{std::numeric_limits<T>::max()};

/** @brief High value as used in kernel currently hard-coded to be below
 * maximum short value of 32767 */
constexpr float kHighValBpKernel{32000.0f};

#if defined(OPTIMIZED_CPU_RUN)
#if defined(FLOAT16_VECTORIZATION)

//specialization of high value value for half type
//that corresponds to max value in float16
template<> inline
const _Float16 kHighValBp<_Float16>(65504);

#endif //FLOAT16_VECTORIZATION
#endif //OPTIMIZED_CPU_RUN

//define specialization for high value in half precision if using CUDA
#if defined(OPTIMIZED_CUDA_RUN)

//set data type used for half-precision with CUDA
#if defined(USE_BFLOAT16_FOR_HALF_PRECISION)
#include <cuda_bf16.h>
//specialization for CUDA bfloat16
template<> inline
const __nv_bfloat16 kHighValBp<__nv_bfloat16>{CUDART_MAX_NORMAL_BF16};
#else
#include <cuda_fp16.h>
//specialization for CUDA bfloat16
template<> inline
const half kHighValBp<half>{CUDART_MAX_NORMAL_FP16};
#endif //USE_BFLOAT16_FOR_HALF_PRECISION

#endif //OPTIMIZED_CUDA_RUN

/**
 * @brief Get number of stereo runs when evaluating implementation
 * Perform less stereo runs if greater number of disparity values
 * since implementation takes longer in those case, so there is likely
 * less variance between runs and therefore less need to have as many runs.
 * 
 * @param disparity_vals 
 * @return Number of Runs to use in benchmarking implementation on stereo set
 */
inline unsigned int NumBpStereoRuns(unsigned int disparity_vals) {
#if defined(FEWER_RUNS_PER_CONFIG)
  //fewer runs if set to use limited parameters/fewer runs
  //for faster processing
  return 3;
#else
  if (disparity_vals > 100) {
    return 7;
  }
  else {
    return 15;
  }
#endif //FEWER_RUNS_PER_CONFIG
}

//by default, optimized memory management and optimized indexing used
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these
//optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized memory management (making the processing more similar to the initial work)
//by setting kUseOptMemManagement to false
//Optimized indexing can be turned off by changing the kOptimizedIndexingSetting value to false
//(not recommended; this slows down processing)
constexpr bool kUseOptMemManagement{true};
constexpr bool kOptimizedIndexingSetting{true};
constexpr bool kAllocateFreeBpMemoryOutsideRuns{true};

//constants for headers for run settings in evaluation
constexpr std::string_view kMemAllocOptHeader{"Memory allocation of all BP data run at or before start of run"};
constexpr std::string_view kMemoryCoalescedBpDataHeader{"BP data arranged for memory coalescence"};
constexpr std::string_view kAllocateFreeMemOutsideRunsHeader{"Memory for BP allocated/freed outside of runs"};

/**
 * @brief Retrieve run settings as a RunData object for output
 * 
 * @return RunData object containing run settings
 */
inline RunData RunSettings()  {
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(
    std::string(kMemAllocOptHeader),
    kUseOptMemManagement);
  curr_run_data.AddDataWHeader(
    std::string(kMemoryCoalescedBpDataHeader),
    kOptimizedIndexingSetting);
  curr_run_data.AddDataWHeader(
    std::string(kAllocateFreeMemOutsideRunsHeader),
    kAllocateFreeBpMemoryOutsideRuns);
  return curr_run_data;
}

/**
 * @brief Inline function to check if data is aligned at x_val_data_start for
 * SIMD loads/stores that require alignment
 * 
 * @param x_val_data_start 
 * @param simd_data_size 
 * @param data_bytes_align_width 
 * @param padded_width_data 
 * @return true if memory aligned at start, false if not
 */
template <RunData_t T>
inline bool MemoryAlignedAtDataStart(
  unsigned int x_val_data_start,
  unsigned int simd_data_size,
  unsigned int data_bytes_align_width,
  unsigned int padded_width_data)
{
  //assuming that the padded checkerboard width divides evenly by
  //beliefprop::NUM_DATA_ALIGN_WIDTH (if that's not the case it's a bug)
  return (((x_val_data_start % simd_data_size) == 0) &&
          (padded_width_data % ((data_bytes_align_width / sizeof(T))) == 0));
}

};

#endif //BP_RUN_UTILS_H
