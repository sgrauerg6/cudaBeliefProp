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

template<class T>
class DisparityMap {
public:
	DisparityMap(unsigned int width, unsigned int height) : width_(width), height_(height), disparity_values_(new T[width*height])
{}

	DisparityMap(unsigned int width, unsigned int height, T* input_disparity_map_vals) : width_(width), height_(height), disparity_values_(new T[width*height])
	{
		std::copy(input_disparity_map_vals, input_disparity_map_vals + (width*height), disparity_values_);
	}

	//destructor
	~DisparityMap()
	{
		delete [] disparity_values_;
	}

	//copy constructor
	DisparityMap(const DisparityMap& in_disp_map)
	{
		width_ = in_disp_map.width_;
		height_ = in_disp_map.height_;
		disparity_values_ = new T[width_*height_];
		std::copy(in_disp_map.disparity_values_, in_disp_map.disparity_values_ + (width_*height_), disparity_values_);
	}

	//copy assignment operator
	DisparityMap& operator=(const DisparityMap& in_disp_map)
	{
		if (this != &in_disp_map)
		{
			delete [] disparity_values_;

			T* new_disparity_values_ = new T[in_disp_map.width_*in_disp_map.height_];
			width_ = in_disp_map.width_;
			height_ = in_disp_map.height_;

			disparity_values_ = new_disparity_values_;
		}

		return *this;
	}

	template<class U>
	const OutputEvaluationResults<T> getOuputComparison(const DisparityMap& disparity_map_to_compare, const OutputEvaluationParameters<U>& evaluation_parameters) const
	{
		OutputEvaluationResults<T> output_evaluation;
		output_evaluation.initializeWithEvalParams(evaluation_parameters);

		for (unsigned int i=0; i<width_*height_; i++)
		{
			int x = i % width_;
			int y = i / width_;
			T dispMapVal = disparity_values_[i];
			T dispMapCompareVal = disparity_map_to_compare.getDisparityValues()[i];
			T absDiff = std::abs(dispMapVal - dispMapCompareVal);

			output_evaluation.totalDispAbsDiffNoMax += absDiff;
			output_evaluation.totalDispAbsDiffWithMax += std::min(absDiff, evaluation_parameters.max_diff_cap_);

			for (float output_diff_threshold : evaluation_parameters.output_diff_thresholds)
			{
				if (absDiff > output_diff_threshold)
				{
					output_evaluation.numSigDiffPixelsAtThresholds[output_diff_threshold]++;
				}
			}
			//std::for_each(evaluation_parameters.output_diff_thresholds_.begin(), evaluation_parameters.output_diff_thresholds_.end(), [dispMapVal, dispMapCompareVal](float threshold) { })
		}

		output_evaluation.averageDispAbsDiffNoMax = output_evaluation.totalDispAbsDiffNoMax / (width_*height_);
		output_evaluation.averageDispAbsDiffWithMax = output_evaluation.totalDispAbsDiffWithMax / (width_*height_);

		for (auto sigDiffPixelAtThresholdMap : output_evaluation.numSigDiffPixelsThreshold)
		{
			output_evaluation.propSigDiffPixelsThreshold[sigDiffPixelAtThresholdMap.first] = sigDiffPixelAtThresholdMap.second / (width_*height_);
		}

		return output_evaluation;
	}

	T* getDisparityValues()
	{
		return disparity_values_;
	}

private:
	T* disparity_values_;
	unsigned int width_;
	unsigned int height_;
};

#endif /* DISPARITYMAP_H_ */
