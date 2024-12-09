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
 * @file BpRunSettings.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BP_RUN_SETTINGS_H
#define BP_RUN_SETTINGS_H

#include <string>
#include <string_view>
#include <array>
#include "RunEval/RunData.h"

/**
 * @brief Namespace for enums, constants, structures, and
 * functions specific to belief propagation processing
 */
namespace beliefprop {

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
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS
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
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS
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
constexpr std::string_view kMemOptLevelHeader{"Memory Optimization Level"};
constexpr std::string_view kIndexingOptLevelHeader{"Indexing Optimization Level"};

/**
 * @brief Retrieve run settings as a RunData object for output
 * 
 * @return RunData 
 */
inline RunData RunSettings()  {
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(std::string(kMemOptLevelHeader),
    std::to_string(kUseOptGPUMemManagement));
  curr_run_data.AddDataWHeader(std::string(kIndexingOptLevelHeader),
    std::to_string(kOptimizedIndexingSetting));
  return curr_run_data;
}

};

#endif //BP_RUN_SETTINGS_H
