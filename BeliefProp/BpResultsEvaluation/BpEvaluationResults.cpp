/*
 * BpEvaluationResults.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "BpEvaluationResults.h"

//initialize evaluation results with evaluation parameters
void BpEvaluationResults::initializeWithEvalParams(const BpEvaluationParameters& evalParams) {
  disparityErrorMax_ = evalParams.max_diff_cap_;
  for (const auto& output_diff_threshold : evalParams.output_diff_thresholds_) {
    numSigDiffPixelsAtThresholds_[output_diff_threshold] = 0;
  }
}

//retrieve evaluation results as RunData for output
RunData BpEvaluationResults::runData() const {
  RunData evalRunData;
  evalRunData.addDataWHeader("Average RMS error", (double)averageDispAbsDiffNoMaxWMax_[0]);
  evalRunData.addDataWHeader("Average RMS error (with disparity error cap at " + std::to_string(disparityErrorMax_) + ")",
                             (double)averageDispAbsDiffNoMaxWMax_[1]);
  for (const auto& propBadPixelsAtThreshold : propSigDiffPixelsAtThresholds_) {
    evalRunData.addDataWHeader("Proportion bad pixels (error less than " + std::to_string(propBadPixelsAtThreshold.first) + ")",
                               (double)propBadPixelsAtThreshold.second);
  }

  return evalRunData;
}
