/*
 * DisparityMap.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef DISPARITYMAP_H_
#define DISPARITYMAP_H_

#include <memory>
#include <algorithm>
#include <string>
#include <iterator>
#include <iostream>
#include "BpImage.h"
#include "../BpOutputEvaluation/OutputEvaluationResults.h"
#include "../BpOutputEvaluation/OutputEvaluationParameters.h"

template<class T>
requires std::is_arithmetic_v<T>
class DisparityMap : public BpImage<T> {
public:
  DisparityMap() : BpImage<T>() {}

  DisparityMap(const std::array<unsigned int, 2>& widthHeight) : BpImage<T>(widthHeight) {}

  DisparityMap(const std::array<unsigned int, 2>& widthHeight, const T* input_disparity_map_vals,
    const unsigned int disparity_map_vals_scale = 1) : BpImage<T>(widthHeight, input_disparity_map_vals)
  {
    std::copy(input_disparity_map_vals, input_disparity_map_vals + this->getTotalPixels(), this->pixels_.get());
    if (disparity_map_vals_scale > 1u) {
      removeScaleFromDisparityVals(disparity_map_vals_scale);
    }
  }

  DisparityMap(const std::string& file_path_disparity_map, const unsigned int disparity_map_vals_scale = 1) : BpImage<T>(file_path_disparity_map)
  {
    if (disparity_map_vals_scale > 1) {
      removeScaleFromDisparityVals(disparity_map_vals_scale);
    }
  }

  const OutputEvaluationResults getOutputComparison(const DisparityMap& disparity_map_to_compare,
    const OutputEvaluationParameters& evaluation_parameters) const;

  void saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor = 1) const;


private:

  void removeScaleFromDisparityVals(const unsigned int disparity_map_vals_scale)
  {
    if (disparity_map_vals_scale > 1) {
      //divide each disparity value by disparity_map_vals_scale
      std::for_each(this->pixels_.get(), this->pixels_.get() + this->getTotalPixels(),
                    [disparity_map_vals_scale](T& disp_val) { disp_val /= disparity_map_vals_scale; });
    }
  }
};

//TODO: look into case where no known disparity in ground truth disparity map (should not be penalized for wrong disparity in that case)
template<class T>
const OutputEvaluationResults DisparityMap<T>::getOutputComparison(
  const DisparityMap& disparity_map_to_compare,
  const OutputEvaluationParameters& evaluation_parameters) const
{
  //initialize output evaluation with evaluation parameters
  OutputEvaluationResults output_evaluation;
  output_evaluation.initializeWithEvalParams(evaluation_parameters);

  //go through each disparity map output pixel and evaluate output
  //against corresponding output in disparity map to compare
  for (unsigned int i = 0; i < this->getTotalPixels(); i++) {
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
    std::for_each(evaluation_parameters.output_diff_thresholds_.begin(), evaluation_parameters.output_diff_thresholds_.end(),
      [absDiff, &output_evaluation](const float threshold) {
        if (absDiff > threshold) {
          output_evaluation.numSigDiffPixelsAtThresholds[threshold]++;
        }
      });
  }

  output_evaluation.averageDispAbsDiffNoMax = output_evaluation.totalDispAbsDiffNoMax / this->getTotalPixels();
  output_evaluation.averageDispAbsDiffWithMax = output_evaluation.totalDispAbsDiffWithMax / this->getTotalPixels();

  //need to cast unsigned ints to float to get proportion of pixels that differ by more than threshold
  std::for_each(output_evaluation.numSigDiffPixelsAtThresholds.begin(), output_evaluation.numSigDiffPixelsAtThresholds.end(),
    [this, &output_evaluation](const auto& sigDiffPixelAtThresholdMap) {
      output_evaluation.propSigDiffPixelsAtThresholds[sigDiffPixelAtThresholdMap.first] =
        ((float)sigDiffPixelAtThresholdMap.second) / ((float)(this->getTotalPixels()));
    });

  return output_evaluation;
}

#endif /* DISPARITYMAP_H_ */
