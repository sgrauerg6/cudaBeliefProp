/*
 * DisparityMap.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "DisparityMap.h"

template<class T>
void DisparityMap<T>::saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor) const {
  //declare and allocate the space for the disparity map image to save
  BpImage<char> movementImageToSave(this->widthHeight_);

  //go though every pixel in the disparity map and compute the intensity value to use in the "disparity map image"
  //by multiplying the pixel disparity value by scale factor
  std::ranges::transform(this->getPointerToPixelsStart(), this->getPointerToPixelsStart() + this->getTotalPixels(),
    movementImageToSave.getPointerToPixelsStart(),
    [scale_factor](const T& currentPixel) -> char {
      return (char)(((float)currentPixel)*((float)scale_factor) + 0.5f);
    });

  movementImageToSave.saveImageAsPgm(disparity_map_file_path);
}

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

    output_evaluation.totalDispAbsDiffNoMax_ += absDiff;
    output_evaluation.totalDispAbsDiffWithMax_ += std::min(absDiff, evaluation_parameters.max_diff_cap_);

    //increment number of pixels that differ greater than threshold for each threshold
    std::ranges::for_each(evaluation_parameters.output_diff_thresholds_,
      [absDiff, &output_evaluation](const float threshold) {
        if (absDiff > threshold) {
          output_evaluation.numSigDiffPixelsAtThresholds_[threshold]++;
        }
      });
  }

  output_evaluation.averageDispAbsDiffNoMax_ = output_evaluation.totalDispAbsDiffNoMax_ / this->getTotalPixels();
  output_evaluation.averageDispAbsDiffWithMax_ = output_evaluation.totalDispAbsDiffWithMax_ / this->getTotalPixels();

  //need to cast unsigned ints to float to get proportion of pixels that differ by more than threshold
  std::ranges::transform(output_evaluation.numSigDiffPixelsAtThresholds_,
    std::inserter(output_evaluation.propSigDiffPixelsAtThresholds_, output_evaluation.propSigDiffPixelsAtThresholds_.end()),
    [this](const auto& sigDiffPixelAtThresholdMap) -> std::pair<float, float> {
      return { sigDiffPixelAtThresholdMap.first, ((float)sigDiffPixelAtThresholdMap.second) / ((float)(this->getTotalPixels())) };
    });

  return output_evaluation;
}

template class DisparityMap<float>;
