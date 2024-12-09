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
 * @file BpFileHandlingConsts.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
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
