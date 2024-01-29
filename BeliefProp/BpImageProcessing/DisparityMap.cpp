/*
 * DisparityMap.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "DisparityMap.h"

template<class T>
void DisparityMap<T>::saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor) const {
  //declare and allocate the space for the movement image to save
  BpImage<char> movementImageToSave(this->widthHeight_);

  //go though every value in the movementBetweenImages data and retrieve the intensity value to use in the resulting "movement image" where minMovementDirection
  //represents 0 intensity and the intensity increases linearly using scaleMovement from minMovementDirection
  std::transform(this->getPointerToPixelsStart(), this->getPointerToPixelsStart() + this->getTotalPixels(),
    movementImageToSave.getPointerToPixelsStart(),
    [this, scale_factor](const T& currentPixel) -> char {
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

template class DisparityMap<float>;
