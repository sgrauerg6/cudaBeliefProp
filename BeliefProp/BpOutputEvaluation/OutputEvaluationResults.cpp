/*
 * OutputEvaluationResults.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "OutputEvaluationResults.h"

//initialize evaluation results with evaluation parameters
void OutputEvaluationResults::initializeWithEvalParams(const OutputEvaluationParameters& evalParams) {
  disparityErrorCap_ = evalParams.max_diff_cap_;
  for (const auto& output_diff_threshold : evalParams.output_diff_thresholds_) {
    numSigDiffPixelsAtThresholds_[output_diff_threshold] = 0;
  }
}

//retrieve evaluation results as RunData for output
RunData OutputEvaluationResults::runData() const {
  RunData evalRunData;
  evalRunData.addDataWHeader("Average RMS error", std::to_string(averageDispAbsDiffNoMax_));
  evalRunData.addDataWHeader("Average RMS error (with disparity error cap at " + std::to_string(disparityErrorCap_) + ")",
                             std::to_string(averageDispAbsDiffNoMax_));
  for (const auto& propBadPixelsAtThreshold : propSigDiffPixelsAtThresholds_) {
    evalRunData.addDataWHeader("Proportion bad pixels (error less than " + std::to_string(propBadPixelsAtThreshold.first) + ")",
                               std::to_string(propBadPixelsAtThreshold.second));
  }

  return evalRunData;
}
