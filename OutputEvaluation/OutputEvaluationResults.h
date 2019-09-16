/*
 * OutputEvaluationResults.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef OUTPUTEVALUATIONRESULTS_H_
#define OUTPUTEVALUATIONRESULTS_H_

#include "OutputEvaluationParameters.h"
#include <map>

template<class T>
class OutputEvaluationResults {
public:
	OutputEvaluationResults() : totalDispAbsDiffNoMax((T)0), totalDispAbsDiffWithMax((T)0), disparityErrorCap((T)9999), averageDispAbsDiffNoMax(0.0f), averageDispAbsDiffWithMax(0.0f) {}
	virtual ~OutputEvaluationResults() {}

	//total value of the absolute difference between the disparity values for all pixels in disparity images 1 and 2 (not including border regions)
	T totalDispAbsDiffNoMax;
	T totalDispAbsDiffWithMax;
	T disparityErrorCap;

	//average absolute difference between the disparity values in disparity images 1 and 2 (not including border regions)
	float averageDispAbsDiffNoMax;
	float averageDispAbsDiffWithMax;

	//proportion of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	//(not including border regions)
	std::map<float, float> propSigDiffPixelsAtThresholds;

	//stores the number of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	std::map<float, unsigned int> numSigDiffPixelsAtThresholds;

	void initializeWithEvalParams(OutputEvaluationParameters<T> evalParams)
	{
		disparityErrorCap = evalParams.max_diff_cap_;
		for (auto output_diff_threshold : evalParams.output_diff_thresholds)
		{
			numSigDiffPixelsAtThresholds[output_diff_threshold] = 0;
		}
	}

	void writeOutputEvaluationResultsToFile(FILE* resultsFile)
	{
		fprintf(resultsFile, "Average RMS error: %f \n", averageDispAbsDiffNoMax);
		fprintf(resultsFile, "Average RMS error (with disparity error cap at %f): %f \n", disparityErrorCap, averageDispAbsDiffWithMax);

		for (auto propBadPixelsAtThreshold : propSigDiffPixelsAtThresholds)
		{
			fprintf(resultsFile, "Proportion bad pixels (error less than %f): %f \n", propBadPixelsAtThreshold.first, propBadPixelsAtThreshold.second);
		}
	}
};

#endif /* OUTPUTEVALUATIONRESULTS_H_ */
