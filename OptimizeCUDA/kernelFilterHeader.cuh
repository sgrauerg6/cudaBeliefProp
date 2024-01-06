/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//Header for the kernel to apply a horizontal/vertical filter to image data

#ifndef KERNEL_FILTER_HEADER_CUH
#define KERNEL_FILTER_HEADER_CUH

#include <cuda_runtime.h>

//checks if the current point is within the image bounds
__device__ bool withinImageBoundsFilter(const unsigned int xVal, const unsigned int yVal,
  const unsigned int width, const unsigned int height);

//kernel to convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//output filtered image stored in floatImagePixels
__global__ void convertUnsignedIntImageToFloat(unsigned int* imagePixelsUnsignedIntToFilter,
  float* floatImagePixels, const unsigned int widthImages, const unsigned int heightImages);

//kernel to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template<typename T>
__global__ void filterImageAcross(T* imagePixelsToFilter, float* filteredImagePixels,
  const unsigned int widthImages, const unsigned int heightImages, float* imageFilter, const unsigned int sizeFilter);

//kernel to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template<typename T>
__global__ void filterImageVertical(T* imagePixelsToFilter, float* filteredImagePixels,
  const unsigned int widthImages, const unsigned int heightImages, float* imageFilter, const unsigned int sizeFilter);

#endif //KERNEL_FILTER_HEADER_CUH
