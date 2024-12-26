/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file DisparityMap.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "DisparityMap.h"

template<class T>
requires std::is_arithmetic_v<T>
void DisparityMap<T>::SaveDisparityMap(const std::string& disparity_map_file_path, unsigned int scale_factor) const {
  //declare and allocate the space for the disparity map image to save
  BpImage<char> disparity_image(this->width_height_);

  //go though every pixel in the disparity map and compute the intensity value to use in the "disparity map image"
  //by multiplying the pixel disparity value by scale factor
  std::ranges::transform(this->PointerToPixelsStart(), this->PointerToPixelsStart() + this->TotalPixels(),
    disparity_image.PointerToPixelsStart(),
    [scale_factor](const T& current_pixel) -> char {
      return (char)(((float)current_pixel)*((float)scale_factor) + 0.5f);
    });

  disparity_image.SaveImageAsPgm(disparity_map_file_path);
}

//TODO: look into case where no known disparity in ground truth disparity map
template<class T>
requires std::is_arithmetic_v<T>
DisparityMapEvaluation DisparityMap<T>::OutputComparison(
  const DisparityMap& disparity_map_to_compare,
  const beliefprop::DisparityMapEvaluationParams& evaluation_parameters) const
{
  //initialize output evaluation with evaluation parameters
  DisparityMapEvaluation output_evaluation;
  output_evaluation.InitializeWithEvalParams(evaluation_parameters);

  //initialize total disparity difference across all pixels with and without max for disparity difference to 0
  std::array<float, 2> total_disp_abs_diff_no_max_w_max{0, 0};

  //go through each disparity map output pixel and evaluate output
  //against corresponding output in disparity map to compare
  for (unsigned int i = 0; i < this->TotalPixels(); i++) {
    //get disparity difference between disparity maps at pixel i
    const T disp_map_val = (this->pixels_.get())[i];
    const T disp_map_compare_val = disparity_map_to_compare.PixelAtPoint(i);
    const T abs_diff = std::abs(disp_map_val - disp_map_compare_val);

    //add disparity difference at pixel to total disparity difference across pixels
    //with and without max disparity difference
    total_disp_abs_diff_no_max_w_max[0] += abs_diff;
    total_disp_abs_diff_no_max_w_max[1] += std::min(abs_diff, evaluation_parameters.max_diff_cap);

    //increment number of pixels that differ greater than threshold for each threshold
    std::ranges::for_each(evaluation_parameters.output_diff_thresholds,
      [abs_diff, &output_evaluation](auto threshold) {
        if (abs_diff > threshold) {
          output_evaluation.num_sig_diff_pixels_at_thresholds_[threshold]++;
        }
      });
  }

  output_evaluation.average_disp_abs_diff_no_max_w_max_[0] =
    total_disp_abs_diff_no_max_w_max[0] / this->TotalPixels();
  output_evaluation.average_disp_abs_diff_no_max_w_max_[1] =
    total_disp_abs_diff_no_max_w_max[1] / this->TotalPixels();

  //need to cast unsigned ints to float to get proportion of pixels that differ by more than threshold
  std::ranges::transform(output_evaluation.num_sig_diff_pixels_at_thresholds_,
    std::inserter(output_evaluation.prop_sig_diff_pixels_at_thresholds_, 
                  output_evaluation.prop_sig_diff_pixels_at_thresholds_.end()),
    [this](const auto& sig_diff_pixel_at_threshold) -> std::pair<float, float> {
      const auto& [threshold, num_sig_diff_pixels_thresh] = sig_diff_pixel_at_threshold;
      return {threshold, ((float)num_sig_diff_pixels_thresh) / ((float)(this->TotalPixels()))};
    });

  return output_evaluation;
}

template class DisparityMap<float>;
