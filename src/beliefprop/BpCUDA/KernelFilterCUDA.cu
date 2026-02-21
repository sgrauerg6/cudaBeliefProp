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
 * @file KernelFilterCUDA.cu
 * @author Scott Grauer-Gray
 * @brief This kernel is used to filter the image with the given filter in the vertical and horizontal directions
 * 
 * @copyright Copyright (c) 2024
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include "BpSharedFuncts/SharedSmoothImageFuncts.h"
#include "RunImp/UtilityFuncts.h"

/**
 * @brief Namespace to define global kernel functions for parallel belief propagation
 * processing using CUDA.
 */
namespace beliefprop_cuda {

/**
 * @brief Kernel to convert the unsigned int pixels to float pixels in an image when
 * smoothing is not desired but the pixels need to be converted to floats
 * Input image is stored as unsigned ints
 * Output filtered image stored in float_image_pixels
 * 
 * @param uint_image_pixels 
 * @param float_image_pixels 
 * @param width_images 
 * @param height_images 
 */
__global__ void convertUnsignedIntImageToFloat(
  unsigned int* uint_image_pixels, float* float_image_pixels,
  unsigned int width_images, unsigned int height_images)
{
  //get x and y indices corresponding to current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  //make sure that (x_val, y_val) is within image bounds
  if (beliefprop::WithinImageBounds(x_val, y_val, width_images, height_images)) {
    //retrieve the float-value of the unsigned int pixel value at the current location
    float_image_pixels[y_val*width_images + x_val] = (float)uint_image_pixels[y_val*width_images + x_val];;
  }
}

/**
 * @brief Kernel to apply a horizontal filter on each pixel of the image in parallel
 * Input image to filter is stored as template parameter type T
 * Output filtered image is array of floats
 * 
 * @tparam T 
 * @param image_to_filter 
 * @param filtered_image 
 * @param width_images 
 * @param height_images 
 * @param image_filter 
 * @param size_filter 
 */
template<BpImData_t T>
__global__ void FilterImageAcross(
  const T* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter)
{
  //get x and y indices corresponding to current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  //make sure that (x_val, y_val) is within image bounds
  if (beliefprop::WithinImageBounds(x_val, y_val, width_images, height_images)) {
    beliefprop::FilterImageAcrossProcessPixel<T>(x_val, y_val, image_to_filter, filtered_image,
      width_images, height_images, image_filter, size_filter);
  }
}

/**
 * @brief Kernel to apply a vertical filter on each pixel of the image in parallel
 * Input image to filter is stored as template parameter type T
 * Output filtered image is array of floats
 * 
 * @tparam T 
 * @param image_to_filter 
 * @param filtered_image 
 * @param width_images 
 * @param height_images 
 * @param image_filter 
 * @param size_filter 
 */
template<BpImData_t T>
__global__ void FilterImageVertical(
  const T* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter)
{
  //get x and y indices corresponding to current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  //make sure that (x_val, y_val) is within image bounds
  if (beliefprop::WithinImageBounds(x_val, y_val, width_images, height_images)) {
    beliefprop::FilterImageVerticalProcessPixel<T>(x_val, y_val, image_to_filter, filtered_image,
      width_images, height_images, image_filter, size_filter);
  }
}

};
