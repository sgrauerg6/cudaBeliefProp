/*
 * DisparityMap.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "DisparityMap.h"

template<class T>
void DisparityMap<T>::saveDisparityMap(const std::string& disparity_map_file_path, unsigned int scale_factor) const {
  //declare and allocate the space for the disparity map image to save
  BpImage<char> disparityImageToSave(this->widthHeight_);

  //go though every pixel in the disparity map and compute the intensity value to use in the "disparity map image"
  //by multiplying the pixel disparity value by scale factor
  std::ranges::transform(this->getPointerToPixelsStart(), this->getPointerToPixelsStart() + this->getTotalPixels(),
    disparityImageToSave.getPointerToPixelsStart(),
    [scale_factor](const T& currentPixel) -> char {
      return (char)(((float)currentPixel)*((float)scale_factor) + 0.5f);
    });

  disparityImageToSave.saveImageAsPgm(disparity_map_file_path);
}

//TODO: look into case where no known disparity in ground truth disparity map (should not be penalized for wrong disparity in that case)
template<class T>
BpEvaluationResults DisparityMap<T>::getOutputComparison(
  const DisparityMap& disparity_map_to_compare,
  const BpEvaluationParameters& evaluation_parameters) const
{
  //initialize output evaluation with evaluation parameters
  BpEvaluationResults output_evaluation;
  output_evaluation.initializeWithEvalParams(evaluation_parameters);

  //initialize total disparity difference across all pixels with and without max for disparity difference to 0
  std::array<float, 2> totalDispAbsDiffNoMaxWMax{0, 0};

  //go through each disparity map output pixel and evaluate output
  //against corresponding output in disparity map to compare
  for (unsigned int i = 0; i < this->getTotalPixels(); i++) {
    //get disparity difference between disparity maps at pixel i
    const T dispMapVal = (this->pixels_.get())[i];
    const T dispMapCompareVal = disparity_map_to_compare.getPixelAtPoint(i);
    const T absDiff = std::abs(dispMapVal - dispMapCompareVal);

    //add disparity difference at pixel to total disparity difference across pixels
    //with and without max disparity difference
    totalDispAbsDiffNoMaxWMax[0] += absDiff;
    totalDispAbsDiffNoMaxWMax[1] += std::min(absDiff, evaluation_parameters.max_diff_cap_);

    //increment number of pixels that differ greater than threshold for each threshold
    std::ranges::for_each(evaluation_parameters.output_diff_thresholds_,
      [absDiff, &output_evaluation](auto threshold) {
        if (absDiff > threshold) {
          output_evaluation.numSigDiffPixelsAtThresholds_[threshold]++;
        }
      });
  }

  output_evaluation.averageDispAbsDiffNoMaxWMax_[0] = totalDispAbsDiffNoMaxWMax[0] / this->getTotalPixels();
  output_evaluation.averageDispAbsDiffNoMaxWMax_[1] = totalDispAbsDiffNoMaxWMax[1] / this->getTotalPixels();

  //need to cast unsigned ints to float to get proportion of pixels that differ by more than threshold
  std::ranges::transform(output_evaluation.numSigDiffPixelsAtThresholds_,
    std::inserter(output_evaluation.propSigDiffPixelsAtThresholds_, output_evaluation.propSigDiffPixelsAtThresholds_.end()),
    [this](const auto& sigDiffPixelAtThresholdMap) -> std::pair<float, float> {
      return { sigDiffPixelAtThresholdMap.first, ((float)sigDiffPixelAtThresholdMap.second) / ((float)(this->getTotalPixels())) };
    });

  return output_evaluation;
}

template class DisparityMap<float>;
