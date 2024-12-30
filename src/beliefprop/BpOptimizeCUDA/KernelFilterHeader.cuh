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
 * @file KernelFilterHeader.cuh
 * @author Scott Grauer-Gray
 * @brief Header for the kernel to apply a horizontal/vertical filter to image data
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef KERNEL_FILTER_HEADER_CUH
#define KERNEL_FILTER_HEADER_CUH

#include <cuda_runtime.h>
#include <type_traits>

/**
 * @brief Namespace to define global kernel functions for parallel belief propagation
 * processing using CUDA.
 * 
 */
namespace beliefprop_cuda {

/**
 * @brief Kernel to convert the unsigned int pixels to float pixels in an image when
 * smoothing is not desired but the pixels need to be converted to floats
 * the input image is stored as unsigned ints
 * output filtered image stored in float_image_pixels
 * 
 * @param uint_image_pixels 
 * @param float_image_pixels 
 * @param width_images 
 * @param height_images 
 */
__global__ void convertUnsignedIntImageToFloat(
  const unsigned int* uint_image_pixels, float* float_image_pixels,
  unsigned int width_images, unsigned int height_images);

/**
 * @brief Kernel to apply a horizontal filter on each pixel of the image in parallel
 * the input image is stored as unsigned ints
 * the output filtered image is returned as an array of floats
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
  const float* image_filter, unsigned int size_filter);

/**
 * @brief Kernel to apply a vertical filter on each pixel of the image in parallel
 * the input image is stored as unsigned ints
 * the output filtered image is returned as an array of floats
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
  const float* image_filter, unsigned int size_filter);

};

#endif //KERNEL_FILTER_HEADER_CUH
