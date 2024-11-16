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
  void initializeWithEvalParams(const BpEvaluationParameters& evalParams);

  //retrieve evaluation results as RunData
  RunData runData() const;

  //total and average value of the absolute difference between the disparity values for all pixels in disparity images 1 and 2
  //with and without maximum disparity difference at each pixel
  std::array<float, 2> averageDispAbsDiffNoMaxWMax_{0, 0};
  float disparityErrorMax_{std::numeric_limits<float>::max()};

  //proportion of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  //(not including border regions)
  std::map<float, float> propSigDiffPixelsAtThresholds_;

  //stores the number of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  std::map<float, unsigned int> numSigDiffPixelsAtThresholds_;
};

#endif /* BPEVALUATIONRESULTS_H_ */
