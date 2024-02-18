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
#include "BpOutputEvaluation/OutputEvaluationResults.h"
#include "BpOutputEvaluation/OutputEvaluationParameters.h"

template<class T>
requires std::is_arithmetic_v<T>
class DisparityMap : public BpImage<T> {
public:
  DisparityMap() : BpImage<T>() {}

  DisparityMap(const std::array<unsigned int, 2>& widthHeight) : BpImage<T>(widthHeight) {}

  DisparityMap(const std::array<unsigned int, 2>& widthHeight, const T* input_disparity_map_vals,
    const unsigned int disparity_map_vals_scale = 1) : BpImage<T>(widthHeight, input_disparity_map_vals)
  {
    std::ranges::copy(input_disparity_map_vals, input_disparity_map_vals + this->getTotalPixels(), this->pixels_.get());
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
      std::ranges::transform(this->pixels_.get(), this->pixels_.get() + this->getTotalPixels(), this->pixels_.get(),
                    [disparity_map_vals_scale](const auto& disp_val) { return (disp_val / disparity_map_vals_scale); });
    }
  }
};

#endif /* DISPARITYMAP_H_ */
