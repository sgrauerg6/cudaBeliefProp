/*
 * BpEvaluationParameters.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef BPEVALUATIONPARAMETERS_H_
#define BPEVALUATIONPARAMETERS_H_

#include <vector>

namespace beliefprop {
  //define the difference in disparity for it to be considered a "significant difference"
  //pixels with difference beyond this are called a "bad pixel" if one of the images is the ground truth
  constexpr float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1 = 0.001;
  constexpr float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2 = 2.01;
  constexpr float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3 = 5.01;
  constexpr float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4 = 10.01;

  //max difference in disparity for evaluation where disparity difference is capped to minimize influence of outliers
  constexpr float MAX_DIFF_CAP = SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4;

  //don't evaluate the disparities that are within the "border"
  //the x-border is from the left/right sides and the
  //y-border are from the top/bottom
  //initialize to the border of the default ground truth disparity
  constexpr unsigned int X_BORDER_SIZE_STEREO_EVAL{0};
  constexpr unsigned int Y_BORDER_SIZE_STEREO_EVAL{0};
}

//structs to store parameters for evaluation of disparity map from stereo processing
struct BpEvaluationParameters {
  std::vector<float> output_diff_thresholds_{
    beliefprop::SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1,
    beliefprop::SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2,
    beliefprop::SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3,
    beliefprop::SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4};
  const float max_diff_cap_{beliefprop::MAX_DIFF_CAP};
  unsigned int x_border_eval_{beliefprop::X_BORDER_SIZE_STEREO_EVAL};
  unsigned int y_border_eval_{beliefprop::Y_BORDER_SIZE_STEREO_EVAL};
};

#endif /* BPEVALUATIONPARAMETERS_H_ */
