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
#include "../imageHelpers.h"
#include <iterator>
#include <iostream>

template<class T>
class DisparityMap {
public:
	DisparityMap() : width_(0), height_(0)
{}

	DisparityMap(const unsigned int width, const unsigned int height) : width_(width), height_(height), disparity_values_(new T[width*height], std::default_delete<T[]>())
{}

	DisparityMap(const unsigned int width, const unsigned int height, const T* input_disparity_map_vals, const unsigned int disparity_map_vals_scale = 1) : width_(width), height_(height), disparity_values_(new T[width*height], std::default_delete<T[]>())
	{
		std::copy(input_disparity_map_vals, input_disparity_map_vals + (width*height), disparity_values_.get());

		if (disparity_map_vals_scale > 1)
		{
			removeScaleFromDisparityVals(disparity_map_vals_scale);
		}
	}

	DisparityMap(const std::string& file_path_disparity_map, const unsigned int disparity_map_vals_scale = 1) : width_(0), height_(0)
	{
		unsigned int* disparity_values_from_file = ImageHelperFunctions::loadImageFromPGM(file_path_disparity_map.c_str(), width_, height_);

		disparity_values_ = std::shared_ptr<T>(new T[width_*height_], std::default_delete<T[]>());;
		for (int i=0; i < (width_*height_); i++)
		{
			(disparity_values_.get())[i] = (T)disparity_values_from_file[i];
		}
		delete [] disparity_values_from_file;

		if (disparity_map_vals_scale > 1)
		{
			removeScaleFromDisparityVals(disparity_map_vals_scale);
		}
	}

	template<class U>
	const OutputEvaluationResults<T> getOuputComparison(const DisparityMap& disparity_map_to_compare, OutputEvaluationParameters<U> evaluation_parameters) const
	{
		OutputEvaluationResults<T> output_evaluation;

		//initials output evaluation with evaluation parameters
		output_evaluation.initializeWithEvalParams(evaluation_parameters);

		for (unsigned int i=0; i<width_*height_; i++)
		{
			//TODO: use x and y with border parameters
			//int x = i % width_;
			//int y = i / width_;
			const T dispMapVal = (disparity_values_.get())[i];
			const T dispMapCompareVal = disparity_map_to_compare.getDisparityValuesAtPoint(i);
			const T absDiff = std::abs(dispMapVal - dispMapCompareVal);

			output_evaluation.totalDispAbsDiffNoMax += absDiff;
			output_evaluation.totalDispAbsDiffWithMax += std::min(absDiff, evaluation_parameters.max_diff_cap_);

			//increment number of pixels that differ greater than threshold for each threshold
			std::for_each(evaluation_parameters.output_diff_thresholds_.begin(), evaluation_parameters.output_diff_thresholds_.end(), [absDiff, &output_evaluation](float threshold)
					{ if (absDiff > threshold) { output_evaluation.numSigDiffPixelsAtThresholds[threshold]++; }	});
		}

		output_evaluation.averageDispAbsDiffNoMax = output_evaluation.totalDispAbsDiffNoMax / (width_*height_);
		output_evaluation.averageDispAbsDiffWithMax = output_evaluation.totalDispAbsDiffWithMax / (width_*height_);

		//need to cast unsigned ints to float to get proportion of pixels that differ by more than threshold
		//typename decltype(output_evaluation.numSigDiffPixelsAtThresholds)::value_type needed to get data type for each mapping; for c++14 can be changed to auto
		std::for_each(output_evaluation.numSigDiffPixelsAtThresholds.begin(), output_evaluation.numSigDiffPixelsAtThresholds.end(),
				[this, &output_evaluation](typename decltype(output_evaluation.numSigDiffPixelsAtThresholds)::value_type sigDiffPixelAtThresholdMap)
				{output_evaluation.propSigDiffPixelsAtThresholds[sigDiffPixelAtThresholdMap.first] = ((float)sigDiffPixelAtThresholdMap.second) / ((float)(this->width_*this->height_));
				});

		return output_evaluation;
	}

	void saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor = 1) const
	{
		float* float_disp_vals = new float[width_*height_];
		for (int i=0; i < (width_*height_); i++)
		{
			float_disp_vals[i] = (float)((disparity_values_.get())[i]);
		}

		ImageHelperFunctions::saveDisparityImageToPGM(disparity_map_file_path.c_str(), scale_factor, float_disp_vals, width_, height_);

		delete [] float_disp_vals;
	}

	std::shared_ptr<T> getDisparityValuesSharedPointer() const
	{
		return disparity_values_;
	}

	const T getDisparityValuesAtPoint(const int x, const int y) const
	{
		return (disparity_values_.get())[y*width_ + x];
	}

	const T getDisparityValuesAtPoint(const int i) const
	{
		return (disparity_values_.get())[i];
	}

	void setDisparityAtPoint(const int x, const int y, const T val)
	{
		(disparity_values_.get())[y*width_ + x] = val;
	}

	void setDisparityAtPoint(const int i, const T val)
	{
		(disparity_values_.get())[i] = val;
	}

private:

	void removeScaleFromDisparityVals(const unsigned int disparity_map_vals_scale)
	{
		if (disparity_map_vals_scale > 1)
		{
			//divide each disparity value by disparity_map_vals_scale
			std::for_each(disparity_values_.get(), disparity_values_.get() + (width_* height_), [disparity_map_vals_scale](T& disp_val) { disp_val /= disparity_map_vals_scale; });
		}
	}

	unsigned int width_;
	unsigned int height_;
	std::shared_ptr<T> disparity_values_;
};

//no need to convert disparity value type if disparity map is of type float
template <>
inline void DisparityMap<float>::saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor) const
{
	ImageHelperFunctions::saveDisparityImageToPGM(disparity_map_file_path.c_str(), scale_factor, disparity_values_.get(), width_, height_);

}

#endif /* DISPARITYMAP_H_ */
