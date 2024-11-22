/*
 * BpFileHandlingConsts.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLINGCONSTS_H_
#define BPFILEHANDLINGCONSTS_H_

#include <string_view>
#include <string>
#include <array>

namespace bp_file_handling
{
  constexpr std::string_view kRefImageName = "refImage";
  constexpr std::string_view kTestImageName = "testImage";
  constexpr std::string_view kInImagePossExtensions[] = {"pgm", "ppm"};
  constexpr std::string_view kGroundTruthDispFile = "groundTruthDisparity.pgm";
  constexpr std::string_view kOutDispImageNameBase = "computedDisparity";
  constexpr std::string_view kBeliefPropDirectoryName = "BeliefProp";
  constexpr std::string_view kStereoSetsDirectoryName = "BpStereoSets";

  #ifdef SMALLER_SETS_ONLY
    constexpr std::array<std::string_view, 2> kBaselineRunDataPathsOptSingleThread{
      "../BeliefProp/BpBaselineRuntimes/baselineRuntimesSmallerSetsOnly.txt",
      "../BeliefProp/BpBaselineRuntimes/singleThreadBaselineRuntimesSmallerSetsOnly.txt"};
  #else
    constexpr std::array<std::string_view, 2> kBaselineRunDataPathsOptSingleThread{
      "../BeliefProp/BpBaselineRuntimes/baselineRuntimes.txt",
      "../BeliefProp/BpBaselineRuntimes/singleThreadBaselineRuntimes.txt"};
  #endif //SMALLER_SETS_ONLY
};

#endif /* BPFILEHANDLINGCONSTS_H_ */
