/*
 * BpStereoParameters.h
 *
 *  Created on: Jun 18, 2019
 *      Author: scott
 */

/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//This header file contains information about the stereo sets used for evaluation of
//the bp implementation.
//Stereo sets are in BpStereoSets folder

#ifndef BP_EVALUATION_STEREO_SETS_H_
#define BP_EVALUATION_STEREO_SETS_H_

#include <string_view>
#include <array>
#include "BpRunProcessing/BpSettings.h"
#include "RunSettingsParams/InputSignature.h"

namespace beliefprop
{
  //headers for image width, height, and stereo set name
  constexpr std::string_view kImageWidthHeader{"Image Width"};
  constexpr std::string_view kImageHeightHeader{"Image Height"};
  constexpr std::string_view kStereoSetHeader{"Stereo Set"};
  
  //structure with stereo set name, disparity count, and scale factor
  //for output disparity map
  struct BpStereoSet {
    const std::string_view name;
    const unsigned int num_disp_vals;
    const unsigned int scale_factor;
  };

  //declare stereo sets to process with name, num disparity values, and scale factor
  //currently conesFullSize is not used
  constexpr std::array<BpStereoSet, 8> kStereoSetsToProcess{
    BpStereoSet{"tsukubaSetHalfSize", 8, 32},
    BpStereoSet{"tsukubaSet", 16, 16},
    BpStereoSet{"venus", 21, 8},
    BpStereoSet{"barn1", 32, 8},
    BpStereoSet{"conesQuarterSize", 64, 4},
    BpStereoSet{"conesHalfSize", 128, 2},
    BpStereoSet{"conesFullSizeCropped", 256, 1},
    BpStereoSet{"conesFullSize", 256, 1}
  };

  //define subsets for evaluating run results on specified inputs
  //the first three stereo sets are labeled as "smallest 3 stereo sets"
  //the last three stereo sets (excluding "conesFullSized") are labeled as "largest 3 stereo sets"
  const std::vector<std::pair<std::string, std::vector<InputSignature>>> kEvalDataSubsets{
    {"smallest 3 stereo sets", 
      {InputSignature({}, 0, {}), InputSignature({}, 1, {}), InputSignature({}, 2, {})}},
    {"largest 3 stereo sets",
      {InputSignature({}, 4, {}), InputSignature({}, 5, {}), InputSignature({}, 6, {})}},
    {"runs w/ disparity count templated",
      {InputSignature({}, {}, true)}},
    {"runs w/ disparity count not templated",
      {InputSignature({}, {}, false)}}};
};

#endif /* BP_EVALUATION_STEREO_SETS_H_ */