/*
 * SharedSmoothImageFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDSMOOTHIMAGEFUNCTS_H_
#define SHAREDSMOOTHIMAGEFUNCTS_H_

#include "RunSettingsEval/RunTypeConstraints.h"
#include "BpImageProcessing/BpImageConstraints.h"
#include "RunImp/RunImpGenFuncts.h"

namespace beliefprop {

//kernel to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture
//the output filtered image is returned as an array of floats
template <BpImData_t T>
ARCHITECTURE_ADDITION inline void FilterImageAcrossProcessPixel(unsigned int x_val, unsigned int y_val,
  const T* image_to_filter, float* filtered_image, unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter)
{
  float filtered_pixel_val = image_filter[0] * ((float)image_to_filter[y_val*width_images + x_val]);

  for (unsigned int i = 1; i < size_filter; i++) {
    filtered_pixel_val +=
      image_filter[i] *
      (((float)image_to_filter[y_val*width_images + (unsigned int)run_imp_util::GetMax((int)x_val - (int)i, 0)]) +
      ((float)image_to_filter[y_val*width_images + run_imp_util::GetMin(x_val + i, width_images - 1)]));
  }

  filtered_image[y_val*width_images + x_val] = filtered_pixel_val;
}

//kernel to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture uint_image_pixelsTexture
//the output filtered image is returned as an array of floats
template <BpImData_t T>
ARCHITECTURE_ADDITION inline void FilterImageVerticalProcessPixel(unsigned int x_val, unsigned int y_val,
  const T* image_to_filter, float* filtered_image, unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter)
{
  float filtered_pixel_val = image_filter[0] * ((float)image_to_filter[y_val*width_images + x_val]);

  for (unsigned int i = 1; i < size_filter; i++) {
    filtered_pixel_val +=
      image_filter[i] *
      ((float)(image_to_filter[(unsigned int)run_imp_util::GetMax((int)y_val - (int)i, 0) * width_images + x_val]) +
      ((float)image_to_filter[run_imp_util::GetMin(y_val + i, height_images - 1) * width_images + x_val]));
  }

  filtered_image[y_val * width_images + x_val] = filtered_pixel_val;
}

}

#endif /* SHAREDSMOOTHIMAGEFUNCTS_H_ */
