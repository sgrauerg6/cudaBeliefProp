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
  constexpr std::string_view IMAGE_WIDTH_HEADER{"Image Width"};
  constexpr std::string_view IMAGE_HEIGHT_HEADER{"Image Height"};
  constexpr std::string_view NUM_DISP_VALS_HEADER{"Num Possible Disparity Values"};
  constexpr std::string_view STEREO_SET_HEADER{"Stereo Set"};
};

#endif //BP_CONSTS_H_
