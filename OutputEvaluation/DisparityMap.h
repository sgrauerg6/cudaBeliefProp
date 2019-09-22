/*
 * DisparityMap.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef DISPARITYMAP_H_
#define DISPARITYMAP_H_

#include <memory>
#include "OutputEvaluationResults.h"
#include "OutputEvaluationParameters.h"
#include <algorithm>
#include <string>
#include <iterator>
#include <iostream>
#include "../ImageDataAndProcessing/BpImage.h"

template<class T>
class DisparityMap : public BpImage<T> {
public:
	DisparityMap() : BpImage<T>()
{}

	DisparityMap(const unsigned int width, const unsigned int height) : BpImage<T>(width, height)
{}

	DisparityMap(const unsigned int width, const unsigned int height, const T* input_disparity_map_vals, const unsigned int disparity_map_vals_scale = 1) : BpImage<T>(width, height, input_disparity_map_vals)
	{
		std::copy(input_disparity_map_vals, input_disparity_map_vals + (width*height), this->pixels_.get());

		if (disparity_map_vals_scale > 1)
		{
			removeScaleFromDisparityVals(disparity_map_vals_scale);
		}
	}

	DisparityMap(const std::string& file_path_disparity_map, const unsigned int disparity_map_vals_scale = 1) : BpImage<T>(file_path_disparity_map)
	{
		if (disparity_map_vals_scale > 1)
		{
			removeScaleFromDisparityVals(disparity_map_vals_scale);
		}
	}

	template<class U>
	const OutputEvaluationResults<T> getOuputComparison(const DisparityMap& disparity_map_to_compare, const OutputEvaluationParameters<U>& evaluation_parameters) const;

	void saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor = 1) const;


private:

	void removeScaleFromDisparityVals(const unsigned int disparity_map_vals_scale)
	{
		if (disparity_map_vals_scale > 1)
		{
			//divide each disparity value by disparity_map_vals_scale
			std::for_each(this->pixels_.get(), this->pixels_.get() + (this->width_* this->height_), [disparity_map_vals_scale](T& disp_val) { disp_val /= disparity_map_vals_scale; });
		}
	}
};

template<class T>
template<class U>
const OutputEvaluationResults<T> DisparityMap<T>::getOuputComparison(
		const DisparityMap& disparity_map_to_compare,
		const OutputEvaluationParameters<U>& evaluation_parameters) const {
	OutputEvaluationResults<T> output_evaluation;

	//initials output evaluation with evaluation parameters
	output_evaluation.initializeWithEvalParams(evaluation_parameters);

	for (unsigned int i = 0; i < this->width_ * this->height_; i++) {
		//TODO: use x and y with border parameters
		//int x = i % width_;
		//int y = i / width_;
		const T dispMapVal = (this->pixels_.get())[i];
		const T dispMapCompareVal = disparity_map_to_compare.getPixelAtPoint(i);
		const T absDiff = std::abs(dispMapVal - dispMapCompareVal);

		output_evaluation.totalDispAbsDiffNoMax += absDiff;
		output_evaluation.totalDispAbsDiffWithMax += std::min(absDiff,
				evaluation_parameters.max_diff_cap_);

		//increment number of pixels that differ greater than threshold for each threshold
		std::for_each(evaluation_parameters.output_diff_thresholds_.begin(),
				evaluation_parameters.output_diff_thresholds_.end(),
				[absDiff, &output_evaluation](float threshold)
				{	if (absDiff > threshold) {output_evaluation.numSigDiffPixelsAtThresholds[threshold]++;}});
	}

	output_evaluation.averageDispAbsDiffNoMax =
			output_evaluation.totalDispAbsDiffNoMax
					/ (this->width_ * this->height_);
	output_evaluation.averageDispAbsDiffWithMax =
			output_evaluation.totalDispAbsDiffWithMax
					/ (this->width_ * this->height_);

	//need to cast unsigned ints to float to get proportion of pixels that differ by more than threshold
	//typename decltype(output_evaluation.numSigDiffPixelsAtThresholds)::value_type needed to get data type for each mapping; for c++14 can be changed to auto
	std::for_each(output_evaluation.numSigDiffPixelsAtThresholds.begin(),
			output_evaluation.numSigDiffPixelsAtThresholds.end(),
			[this, &output_evaluation](typename decltype(output_evaluation.numSigDiffPixelsAtThresholds)::value_type sigDiffPixelAtThresholdMap)
			{	output_evaluation.propSigDiffPixelsAtThresholds[sigDiffPixelAtThresholdMap.first] = ((float)sigDiffPixelAtThresholdMap.second) / ((float)(this->width_*this->height_));
			});

	return output_evaluation;
}

#endif /* DISPARITYMAP_H_ */
