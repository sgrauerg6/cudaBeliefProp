/*
 * BpEvaluationParameters.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef BPEVALUATIONPARAMETERS_H_
#define BPEVALUATIONPARAMETERS_H_

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
 * @brief Structs to store parameters for evaluation of disparity map from stereo processing
 * 
 */
struct BpEvaluationParameters {
  /**
   * @brief Evaluation done at multiple difference thresholds
   * 
   */
  const std::vector<float> output_diff_thresholds{
    beliefprop::kDisparityDiffThresholds.cbegin(),
    beliefprop::kDisparityDiffThresholds.cend()};
  const float max_diff_cap{beliefprop::kMaxDiffCap};
};

#endif /* BPEVALUATIONPARAMETERS_H_ */
