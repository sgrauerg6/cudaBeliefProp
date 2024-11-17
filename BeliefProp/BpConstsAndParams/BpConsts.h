/*
 * BpConsts.h
 *
 *  Created on: March 2, 2024
 *      Author: scott
 */

#ifndef BP_CONSTS_H_
#define BP_CONSTS_H_

#include <string_view>

namespace belief_prop {
  //constants for headers in belief propagation input and results
  constexpr std::string_view kImageWidthHeader{"Image Width"};
  constexpr std::string_view kImageHeightHeader{"Image Height"};
  constexpr std::string_view kNumDispValsHeader{"Num Possible Disparity Values"};
  constexpr std::string_view kStereoSetHeader{"Stereo Set"};
};

#endif //BP_CONSTS_H_
