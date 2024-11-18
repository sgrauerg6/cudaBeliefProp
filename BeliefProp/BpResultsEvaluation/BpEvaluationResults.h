/*
 * BpEvaluationResults.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef BPEVALUATIONRESULTS_H_
#define BPEVALUATIONRESULTS_H_

#include <map>
#include <iostream>
#include <numeric>
#include "RunSettingsEval/RunData.h"
#include "BpEvaluationParameters.h"

//class to store stereo processing evaluation results
class BpEvaluationResults {
public:
  //initialize evaluation results with evaluation parameters
  void InitializeWithEvalParams(const BpEvaluationParameters& eval_params);

  //retrieve evaluation results as RunData
  RunData AsRunData() const;

  //total and average value of the absolute difference between the disparity values for all pixels in disparity images 1 and 2
  //with and without maximum disparity difference at each pixel
  std::array<float, 2> average_disp_abs_diff_no_max_w_max_{0, 0};
  float disparity_error_max_{std::numeric_limits<float>::max()};

  //proportion of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  //(not including border regions)
  std::map<float, float> prop_sig_diff_pixels_at_thresholds_;

  //stores the number of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  std::map<float, unsigned int> num_sig_diff_pixels_at_thresholds_;
};

#endif /* BPEVALUATIONRESULTS_H_ */
