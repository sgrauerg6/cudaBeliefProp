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
 * @file SmoothImage.h
 * @author Scott Grauer-Gray
 * @brief Declares class for smoothing the images before running BP
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef SMOOTH_IMAGE_HOST_HEADER_H
#define SMOOTH_IMAGE_HOST_HEADER_H

#include <cmath>
#include <memory>
#include <utility>
#include <algorithm>
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpRunProcessing/ParallelParamsBp.h"
#include "BpImage.h"

//don't smooth input images if kSigmaBp below this
constexpr float kMinSigmaValSmooth{0.1f};

//parameter for smoothing
constexpr float kWidthSigma1{4.0f};

/**
 * @brief Class for smoothing the images before running BP.
 * Smoothing image always uses float data type.
 * 
 */
class SmoothImage
{
public:
  SmoothImage(const ParallelParams& parallel_params) : parallel_params_{parallel_params} {}

  /**
   * @brief Function to use the image filter to apply a guassian filter to the a single images
   * input images have each pixel stored as an unsigned int (value between 0 and 255 assuming 8-bit grayscale image used)
   * output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigma
   * normalize mask so it integrates to one
   * 
   * @param in_image 
   * @param sigma 
   * @param smoothed_image 
   */
  virtual void operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image) const = 0;

protected:
  /**
   * @brief Create a Gaussian filter from a sigma value
   * 
   * @param sigma 
   * @return std::pair<std::vector<float>, unsigned int> 
   */
  std::pair<std::vector<float>, unsigned int> MakeFilter(float sigma) const;

  /**
   * @brief Parallel parameters to use parallel operations
   * (number of threads on CPU / thread block config in CUDA)
   * 
   */
  const ParallelParams& parallel_params_;

private:
  /**
   * @brief Normalize filter mask so it integrates to one
   * 
   * @param filter 
   * @param size_filter 
   */
  void NormalizeFilter(std::vector<float>& filter, unsigned int size_filter) const;
};

#endif //SMOOTH_IMAGE_HOST_HEADER_H
