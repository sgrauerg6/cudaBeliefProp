/*
 * BpRunSettings.h
 *
 *  Created on: Nov 19, 2024
 *      Author: scott
 */

#ifndef BP_RUN_SETTINGS_H
#define BP_RUN_SETTINGS_H

#include <string>
#include <string_view>
#include <array>
#include "RunEval/RunData.h"

namespace beliefprop {

//number of belief propagation stereo runs of same image set
//fewer runs if using limited parameters for faster processing
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS
  constexpr unsigned int kNumBpStereoRuns = 3;
#else
  constexpr unsigned int kNumBpStereoRuns = 15;
#endif //LIMITED_TEST_PARAMS_FEWER_RUNS

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

  //retrieve run settings as a RunData object for output
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
