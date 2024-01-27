/*
 * BpFileHandlingConsts.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLINGCONSTS_H_
#define BPFILEHANDLINGCONSTS_H_

#include <string_view>
#include <array>

namespace bp_file_handling
{
  constexpr std::string_view REF_IMAGE_NAME = "refImage";
  constexpr std::string_view TEST_IMAGE_NAME = "testImage";
  constexpr std::string_view IN_IMAGE_POSS_EXTENSIONS[] = {"pgm", "ppm"};
  constexpr std::string_view GROUND_TRUTH_DISP_FILE = "groundTruthDisparity.pgm";
  constexpr std::string_view OUT_DISP_IMAGE_NAME_BASE = "computedDisparity";
  constexpr std::string_view BELIEF_PROP_DIRECTORY_NAME = "BeliefProp";
  constexpr std::string_view STEREO_SETS_DIRECTORY_NAME = "BpStereoSets";

  #ifdef SMALLER_SETS_ONLY
    constexpr std::array<std::string_view, 2> BASELINE_RUN_DATA_PATHS_OPT_SINGLE_THREAD{
      "../BeliefProp/BpBaselineRuntimes/baselineRuntimesSmallerSetsOnly.txt",
      "../BeliefProp/BpBaselineRuntimes/singleThreadBaselineRuntimesSmallerSetsOnly.txt"};
  #else
    constexpr std::array<std::string_view, 2> BASELINE_RUN_DATA_PATHS_OPT_SINGLE_THREAD{
      "../BeliefProp/BpBaselineRuntimes/baselineRuntimes.txt",
      "../BeliefProp/BpBaselineRuntimes/singleThreadBaselineRuntimes.txt"};
  #endif //SMALLER_SETS_ONLY
}

#endif /* BPFILEHANDLINGCONSTS_H_ */
