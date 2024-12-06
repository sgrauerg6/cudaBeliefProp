/*
 * DisparityMap.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef DISPARITYMAP_H_
#define DISPARITYMAP_H_

#include <algorithm>
#include <string>
#include <ranges>
#include "BpImage.h"
#include "BpResultsEvaluation/BpEvaluationResults.h"
#include "BpResultsEvaluation/BpEvaluationParameters.h"

/**
 * @brief Class to define disparity map image that is output from bp processing
 * 
 * @tparam T 
 */
template<class T>
requires std::is_arithmetic_v<T>
class DisparityMap : public BpImage<T> {
public:
  DisparityMap() : BpImage<T>() {}

  DisparityMap(const std::array<unsigned int, 2>& width_height) : BpImage<T>(width_height) {}

  DisparityMap(const std::array<unsigned int, 2>& width_height, const T* input_disparity_map_vals,
    unsigned int disparity_map_vals_scale = 1) : BpImage<T>(width_height, input_disparity_map_vals)
  {
    std::ranges::copy(input_disparity_map_vals,
      input_disparity_map_vals + this->TotalPixels(),
      this->pixels_.get());

    if (disparity_map_vals_scale > 1u) {
      RemoveScaleFromDisparity_vals(disparity_map_vals_scale);
    }
  }

  DisparityMap(const std::string& file_path_disparity_map, unsigned int disparity_map_vals_scale = 1) : 
    BpImage<T>(file_path_disparity_map)
  {
    if (disparity_map_vals_scale > 1) {
      RemoveScaleFromDisparity_vals(disparity_map_vals_scale);
    }
  }

  BpEvaluationResults OutputComparison(const DisparityMap& disparity_map_to_compare,
    const BpEvaluationParameters& evaluation_parameters) const;

  void SaveDisparityMap(const std::string& disparity_map_file_path, unsigned int scale_factor = 1) const;

private:
  void RemoveScaleFromDisparity_vals(unsigned int disparity_map_vals_scale)
  {
    if (disparity_map_vals_scale > 1) {
      //divide each disparity value by disparity_map_vals_scale
      std::ranges::transform(this->pixels_.get(), this->pixels_.get() + this->TotalPixels(),
        this->pixels_.get(),
        [disparity_map_vals_scale](const auto& disp_val) { 
          return (disp_val / disparity_map_vals_scale); });
    }
  }
};

#endif /* DISPARITYMAP_H_ */
