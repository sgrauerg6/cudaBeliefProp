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
 * @file SmoothImageCPU.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of SmoothImage for smoothing images in the
 * optimized CPU implementation
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef SMOOTHIMAGECPU_H_
#define SMOOTHIMAGECPU_H_

#include <utility>
#include <memory>
#include "BpImageProcessing/SmoothImage.h"
#include "BpImageProcessing/BpImageConstraints.h"
#include "RunSettingsParams/ParallelParams.h"

/**
 * @brief Child class of SmoothImage for smoothing images in the optimized CPU implementation
 * 
 */
class SmoothImageCPU final : public SmoothImage {
public:
  SmoothImageCPU(const ParallelParams& opt_cpu_params) : SmoothImage(opt_cpu_params) {}

  //function to use the CPU image filter to apply a guassian filter to the a single images
  //input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
  //output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigma
  void operator()(const BpImage<unsigned int>& in_image, const float sigma, float* smoothed_image) const override;

private:
  //convert the unsigned int pixels to float pixels in an image when
  //smoothing is not desired but the pixels need to be converted to floats
  //the input image is stored as unsigned ints in the texture uint_image_pixels
  //output filtered image stored in float_image_pixels
  void ConvertUnsignedIntImageToFloatCPU(
    const unsigned int* uint_image_pixels, float* float_image_pixels,
    unsigned int width_images, unsigned int height_images,
    const ParallelParams& opt_cpu_params) const;

  //apply a horizontal filter on each pixel of the image in parallel
  template<BpImData_t U>
  void FilterImageAcrossCPU(
    const U* image_to_filter, float* filtered_image,
    unsigned int width_images, unsigned int height_images,
    const float* image_filter, unsigned int size_filter,
    const ParallelParams& opt_cpu_params) const;

  //apply a vertical filter on each pixel of the image in parallel
  template<BpImData_t U>
  void FilterImageVerticalCPU(
    const U* image_to_filter, float* filtered_image,
    unsigned int width_images, unsigned int height_images,
    const float* image_filter, unsigned int size_filter,
    const ParallelParams& opt_cpu_params) const;
};

#endif /* SMOOTHIMAGECPU_H_ */
