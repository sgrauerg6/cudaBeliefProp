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
 * @file DisparityMap.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of BpImage to define disparity map image that
 * is output from bp processing
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef DISPARITYMAP_H_
#define DISPARITYMAP_H_

#include <algorithm>
#include <string>
#include <ranges>
#include "BpImage.h"
#include "BpResultsEvaluation/DisparityMapEvaluation.h"

/**
 * @brief Child class of BpImage to define disparity map image that is output
 * from bp processing
 * 
 * @tparam T 
 */
template<class T>
requires std::is_arithmetic_v<T>
class DisparityMap : public BpImage<T> {
public:
  DisparityMap() : BpImage<T>() {}

  explicit DisparityMap(
    const std::array<unsigned int, 2>& width_height) :
      BpImage<T>(width_height) {}

  explicit DisparityMap(
    const std::array<unsigned int, 2>& width_height,
    const T* input_disparity_map_vals,
    unsigned int disparity_map_vals_scale = 1) : 
      BpImage<T>(width_height, input_disparity_map_vals)
  {
    std::ranges::copy(input_disparity_map_vals,
      input_disparity_map_vals + this->TotalPixels(),
      this->pixels_.get());

    if (disparity_map_vals_scale > 1u) {
      RemoveScaleFromDisparity_vals(disparity_map_vals_scale);
    }
  }

  explicit DisparityMap(
    const std::string& file_path_disparity_map,
    unsigned int disparity_map_vals_scale = 1) : 
      BpImage<T>(file_path_disparity_map)
  {
    if (disparity_map_vals_scale > 1) {
      RemoveScaleFromDisparity_vals(disparity_map_vals_scale);
    }
  }

  DisparityMapEvaluation OutputComparison(
    const DisparityMap& disparity_map_to_compare,
    const beliefprop::DisparityMapEvaluationParams& evaluation_parameters) const;

  void SaveDisparityMap(
    const std::string& disparity_map_file_path,
    unsigned int scale_factor = 1) const;

private:
  void RemoveScaleFromDisparity_vals(unsigned int disparity_map_vals_scale)
  {
    if (disparity_map_vals_scale > 1) {
      //divide each disparity value by disparity_map_vals_scale
      std::ranges::transform(
        this->pixels_.get(),
        this->pixels_.get() + this->TotalPixels(),
        this->pixels_.get(),
        [disparity_map_vals_scale](const auto& disp_val) { 
          return (disp_val / disparity_map_vals_scale); });
    }
  }
};

#endif /* DISPARITYMAP_H_ */
