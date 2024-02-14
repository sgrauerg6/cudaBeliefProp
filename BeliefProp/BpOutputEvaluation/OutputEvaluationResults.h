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
#include <numeric>
#include "RunSettingsEval/RunData.h"
#include "OutputEvaluationParameters.h"

//class to store stereo processing evaluation results
class OutputEvaluationResults {
public:
  //initialize evaluation results with evaluation parameters
  void initializeWithEvalParams(const OutputEvaluationParameters& evalParams);

  //retrieve evaluation results as RunData for output
  RunData runData() const;

  //total value of the absolute difference between the disparity values for all pixels in disparity images 1 and 2 (not including border regions)
  float totalDispAbsDiffNoMax_{0};
  float totalDispAbsDiffWithMax_{0};
  float disparityErrorCap_{std::numeric_limits<float>::max()};

  //average absolute difference between the disparity values in disparity images 1 and 2 (not including border regions)
  float averageDispAbsDiffNoMax_{0};
  float averageDispAbsDiffWithMax_{0};

  //proportion of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  //(not including border regions)
  std::map<float, float> propSigDiffPixelsAtThresholds_;

  //stores the number of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
  std::map<float, unsigned int> numSigDiffPixelsAtThresholds_;
};

#endif /* OUTPUTEVALUATIONRESULTS_H_ */
