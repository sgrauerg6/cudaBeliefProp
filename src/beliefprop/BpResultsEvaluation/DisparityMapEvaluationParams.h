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
 * @file DisparityMapEvaluationParams.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef DISPARITY_MAP_EVALUATION_PARAMS_H_
#define DISPARITY_MAP_EVALUATION_PARAMS_H_

#include <vector>
#include <array>

namespace beliefprop {
  /**
   * @brief Difference thresholds in output disparity for a computed disparity at a pixel
   * to be considered a "bad pixel" when compared to the ground truth in the evaluation
   * 
   */
  constexpr std::array<float, 4> kDisparityDiffThresholds{
    0.001, 2.01, 5.01, 10.01};

  /**
   * @brief Max difference in disparity for evaluation where disparity difference for each pixel is capped to minimize influence of outliers
   * in the average difference across all pixels
   * 
   */
  constexpr float kMaxDiffCap{
    kDisparityDiffThresholds[std::size(kDisparityDiffThresholds) - 1]};
}

/**
 * @brief Struct to store parameters for evaluation of disparity map from stereo processing
 * 
 */
struct DisparityMapEvaluationParams {
  /**
   * @brief Evaluation done at multiple difference thresholds
   * 
   */
  const std::vector<float> output_diff_thresholds{
    beliefprop::kDisparityDiffThresholds.cbegin(),
    beliefprop::kDisparityDiffThresholds.cend()};
  const float max_diff_cap{beliefprop::kMaxDiffCap};
};

#endif /* DISPARITY_MAP_EVALUATION_PARAMS_H_ */
