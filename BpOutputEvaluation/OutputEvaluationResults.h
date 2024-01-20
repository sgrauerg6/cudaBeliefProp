/*
 * OutputEvaluationResults.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef OUTPUTEVALUATIONRESULTS_H_
#define OUTPUTEVALUATIONRESULTS_H_

#include <map>
#include <iostream>
#include "../RunSettingsEval/RunData.h"
#include "OutputEvaluationParameters.h"

class OutputEvaluationResults {
public:
  OutputEvaluationResults() : totalDispAbsDiffNoMax(0), totalDispAbsDiffWithMax(0), disparityErrorCap(9999),
                              averageDispAbsDiffNoMax(0.0f), averageDispAbsDiffWithMax(0.0f) {}
  virtual ~OutputEvaluationResults() {}

  //total value of the absolute difference between the disparity values for all pixels in disparity images 1 and 2 (not including border regions)
  float totalDispAbsDiffNoMax;
  float totalDispAbsDiffWithMax;
  float disparityErrorCap;

  //average absolute difference between the disparity values in disparity images 1 and 2 (not including border regions)
  float averageDispAbsDiffNoMax;
  float averageDispAbsDiffWithMax;

  //proportion of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  //(not including border regions)
  std::map<float, float> propSigDiffPixelsAtThresholds;

  //stores the number of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  std::map<float, unsigned int> numSigDiffPixelsAtThresholds;

  void initializeWithEvalParams(const OutputEvaluationParameters& evalParams) {
    disparityErrorCap = evalParams.max_diff_cap_;
    for (const auto& output_diff_threshold : evalParams.output_diff_thresholds_) {
      numSigDiffPixelsAtThresholds[output_diff_threshold] = 0;
    }
  }

  RunData runData() const {
    RunData evalRunData;
    evalRunData.addDataWHeader("Average RMS error", std::to_string(averageDispAbsDiffNoMax));
    evalRunData.addDataWHeader("Average RMS error (with disparity error cap at " + std::to_string(disparityErrorCap) + ")",
                               std::to_string(averageDispAbsDiffNoMax));
    for (const auto& propBadPixelsAtThreshold : propSigDiffPixelsAtThresholds) {
      evalRunData.addDataWHeader("Proportion bad pixels (error less than " + std::to_string(propBadPixelsAtThreshold.first) + ")",
                                 std::to_string(propBadPixelsAtThreshold.second));
    }

    return evalRunData;
  }

  inline friend std::ostream& operator<<(std::ostream& os, const OutputEvaluationResults& results)
  {
    os << "Average RMS error: " << results.averageDispAbsDiffNoMax << std::endl;
    os << "Average RMS error (with disparity error cap at " << results.disparityErrorCap << "): " <<  results.averageDispAbsDiffWithMax << std::endl;

    for (const auto& propBadPixelsAtThreshold : results.propSigDiffPixelsAtThresholds) {
      os << "Proportion bad pixels (error less than " << propBadPixelsAtThreshold.first << "): " << propBadPixelsAtThreshold.second << std::endl;
    }

    return os;
  }
};

#endif /* OUTPUTEVALUATIONRESULTS_H_ */
