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
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BP_RUN_UTILS_H
#define BP_RUN_UTILS_H

#include <string>
#include <string_view>
#include <array>
#include "RunEval/RunData.h"

/**
 * @brief Namespace for enums, constants, structures, and
 * functions specific to belief propagation processing
 */
namespace beliefprop {

/** @brief Float value of "infinity" that works with half-precision */
constexpr float kInfBp{65504};

/**
 * @brief Get number of stereo runs when evaluating implementation
 * Perform less stereo runs if greater number of disparity values
 * since implementation takes longer in those case, so there is likely
 * less variance between runs and therefore less need to have as many runs.
 * 
 * @param disparity_vals 
 * @return unsigned int 
 */
inline unsigned int NumBpStereoRuns(unsigned int disparity_vals) {
#ifdef FEWER_RUNS_PER_CONFIG
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

//by default, optimized GPU memory management and optimized indexing used
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these
//optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work)
//by setting kUseOptGPUMemManagement to false
//Optimized indexing can be turned off by changing the kOptimizedIndexingSetting value to false
//(not recommended; this slows down processing)
constexpr bool kUseOptGPUMemManagement{true};
constexpr bool kOptimizedIndexingSetting{true};
constexpr bool kAllocateFreeBpMemoryOutsideRuns{true};

//constants for headers for run settings in evaluation
constexpr std::string_view kMemAllocOptHeader{"Memory allocation of all BP data run at or before start of run"};
constexpr std::string_view kMemoryCoalescedBpDataHeader{"BP data arranged for memory coalescence"};
constexpr std::string_view kAllocateFreeMemOutsideRunsHeader{"Memory for BP allocated/freed outside of runs"};

/**
 * @brief Retrieve run settings as a RunData object for output
 * 
 * @return RunData 
 */
inline RunData RunSettings()  {
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(
    std::string(kMemAllocOptHeader),
    kUseOptGPUMemManagement);
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
 * @param num_data_align_width 
 * @param divPaddedChBoardWidthForAlign 
 * @return true 
 * @return false 
 */
inline bool MemoryAlignedAtDataStart(
  unsigned int x_val_data_start,
  unsigned int simd_data_size,
  unsigned int num_data_align_width,
  unsigned int divPaddedChBoardWidthForAlign)
{
  //assuming that the padded checkerboard width divides evenly by beliefprop::NUM_DATA_ALIGN_WIDTH (if that's not the case it's a bug)
  return (((x_val_data_start % simd_data_size) == 0) && ((num_data_align_width % divPaddedChBoardWidthForAlign) == 0));
}

};

#endif //BP_RUN_UTILS_H
