/*
 * DisparityMapEvaluation.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "DisparityMapEvaluation.h"

//initialize evaluation results with evaluation parameters
void DisparityMapEvaluation::InitializeWithEvalParams(const DisparityMapEvaluationParams& eval_params) {
  disparity_error_max_ = eval_params.max_diff_cap;
  for (const auto& output_diff_threshold : eval_params.output_diff_thresholds) {
    num_sig_diff_pixels_at_thresholds_[output_diff_threshold] = 0;
  }
}

//retrieve evaluation results as RunData for output
RunData DisparityMapEvaluation::AsRunData() const {
  RunData evalRunData;
  evalRunData.AddDataWHeader(std::string(beliefprop::kAvgRMSErrorHeader),
    (double)average_disp_abs_diff_no_max_w_max_[0]);
  evalRunData.AddDataWHeader(
    std::string(beliefprop::kAvgRMSErrorHeader) + " (with disparity error cap at " + 
      std::to_string(disparity_error_max_) + ")",
    (double)average_disp_abs_diff_no_max_w_max_[1]);
  for (const auto& propBadPixelsAtThreshold : prop_sig_diff_pixels_at_thresholds_) {
    evalRunData.AddDataWHeader("Proportion bad pixels (error less than " + std::to_string(propBadPixelsAtThreshold.first) + ")",
                               (double)propBadPixelsAtThreshold.second);
  }

  return evalRunData;
}
