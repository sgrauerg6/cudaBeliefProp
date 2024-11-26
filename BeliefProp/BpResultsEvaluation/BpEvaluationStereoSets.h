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
#include "RunSettingsEval/EvalInputSignature.h"

namespace beliefprop
{
  constexpr std::string_view kImageWidthHeader{"Image Width"};
  constexpr std::string_view kImageHeightHeader{"Image Height"};
  constexpr std::string_view kStereoSetHeader{"Stereo Set"};
  
  struct BpStereoSet {
    const std::string_view name;
    const unsigned int num_disp_vals;
    const unsigned int scale_factor;
  };

  //declare stereo sets to process with name, num disparity values, and scale factor
  //currently conesFullSize is not used
  //the first three stereo sets are labeled as "smallest 3 stereo sets"
  //the last three stereo sets (excluding "conesFullSized") are labeled as "largest 3 stereo sets"
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

  //constants defined indices of smallest and largest three stereo sets
  //used in evaluation
  constexpr std::array<std::array<std::size_t, 3>, 2> kSmallLargeStereoSetsEvalNums{
    std::array<std::size_t, 3>{0, 1, 2},
    std::array<std::size_t, 3>{4, 5, 6}};

  //input signature for small stereo sets in evaluation
  //currently only evaluating small stereo sets by themselves with float data type
  const std::vector<EvalInputSignature> kSmallStereoSetsInputSigs{
    EvalInputSignature(sizeof(float), 0, false), EvalInputSignature(sizeof(float), 0, true),
    EvalInputSignature(sizeof(float), 1, false), EvalInputSignature(sizeof(float), 1, true),
    EvalInputSignature(sizeof(float), 2, false), EvalInputSignature(sizeof(float), 2, true)};

  //input signature for large stereo sets in evaluation
  //currently only evaluating large stereo sets by themselves with float data type
  const std::vector<EvalInputSignature> kLargeStereoSetsInputSigs{
    EvalInputSignature(sizeof(float), 4, false), EvalInputSignature(sizeof(float), 4, true),
    EvalInputSignature(sizeof(float), 5, false), EvalInputSignature(sizeof(float), 5, true),
    EvalInputSignature(sizeof(float), 6, false), EvalInputSignature(sizeof(float), 6, true)};

  //strings for description of smallest and largest stereo sets in evaluation
  constexpr std::array<std::string_view, 2> kSmallLargeStereoSetsEvalStr{
    "smallest 3 stereo sets", "largest 3 stereo sets"};
};

#endif /* BP_EVALUATION_STEREO_SETS_H_ */
