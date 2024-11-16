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
  //difference thresholds in output disparity for a computed disparity at a pixel
  //to be considered a "bad pixel" when compared to the ground truth in the evaluation
  constexpr std::array<float, 4> SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLDS{
    0.001, 2.01, 5.01, 10.01};

  //max difference in disparity for evaluation where disparity difference for each pixel is capped to minimize influence of outliers
  //in the average difference across all pixels
  constexpr float MAX_DIFF_CAP{
    SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLDS[std::size(SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLDS) - 1]};
}

//structs to store parameters for evaluation of disparity map from stereo processing
struct BpEvaluationParameters {
  //evaluation done at multiple difference thresholds
  const std::vector<float> output_diff_thresholds_{
    beliefprop::SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLDS.begin(),
    beliefprop::SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLDS.end()};
  const float max_diff_cap_{beliefprop::MAX_DIFF_CAP};
};

#endif /* BPEVALUATIONPARAMETERS_H_ */
