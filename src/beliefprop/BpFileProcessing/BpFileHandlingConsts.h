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
  constexpr std::string_view kBaselineRunDataPath{
    "../BeliefProp/ImpResults/RunResults/AMDRome48Cores_RunResults.csv"};
  constexpr std::string_view kBaselineRunDesc{"AMD Rome (48 Cores)"};
};

#endif /* BPFILEHANDLINGCONSTS_H_ */
