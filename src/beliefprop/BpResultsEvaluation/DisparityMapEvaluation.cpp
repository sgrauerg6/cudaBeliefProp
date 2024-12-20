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
 * @file DisparityMapEvaluation.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "DisparityMapEvaluation.h"

//initialize evaluation results with evaluation parameters
void DisparityMapEvaluation::InitializeWithEvalParams(const beliefprop::DisparityMapEvaluationParams& eval_params) {
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
  for (const auto& [threshold, prop_bad_pixels] : prop_sig_diff_pixels_at_thresholds_) {
    evalRunData.AddDataWHeader("Proportion bad pixels (error less than " + std::to_string(threshold) + ")",
                               (double)prop_bad_pixels);
  }

  return evalRunData;
}
