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

//This kernel is used to filter the image with the given filter in the vertical and horizontal directions

#include "kernelFilterHeader.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include "../BpConstsAndParams/bpStereoCudaParameters.h"
#include "../SharedFuncts/SharedSmoothImageFuncts.h"

//checks if the current point is within the image bounds
__device__ bool withinImageBoundsFilter(
  const unsigned int xVal, const unsigned int yVal,
  const unsigned int width, const unsigned int height)
{
  //xVal and yVal unsigned so no need to compare with zero
  return ((xVal < width) && (yVal < height));
}


//kernel to convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//output filtered image stored in floatImagePixels
__global__ void convertUnsignedIntImageToFloat(
  unsigned int* imagePixelsUnsignedIntToFilter, float* floatImagePixels,
  const unsigned int widthImages, const unsigned int heightImages)
{
  //get x and y indices corresponding to current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  //make sure that (xVal, yVal) is within image bounds
  if (withinImageBoundsFilter(xVal, yVal, widthImages, heightImages)) {
    //retrieve the float-value of the unsigned int pixel value at the current location
    floatImagePixels[yVal*widthImages + xVal] = (float)imagePixelsUnsignedIntToFilter[yVal*widthImages + xVal];;
  }
}


//kernel to apply a horizontal filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
template<BpImData_t T>
__global__ void filterImageAcross(
  T* imagePixelsToFilter, float* filteredImagePixels,
  const unsigned int widthImages, const unsigned int heightImages,
  float* imageFilter, const unsigned int sizeFilter)
{
  //get x and y indices corresponding to current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  //make sure that (xVal, yVal) is within image bounds
  if (withinImageBoundsFilter(xVal, yVal, widthImages, heightImages)) {
    filterImageAcrossProcessPixel<T>(xVal, yVal, imagePixelsToFilter, filteredImagePixels,
      widthImages, heightImages, imageFilter, sizeFilter);
  }
}


//kernel to apply a vertical filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
template<BpImData_t T>
__global__ void filterImageVertical(
  T* imagePixelsToFilter, float* filteredImagePixels,
  const unsigned int widthImages, const unsigned int heightImages,
  float* imageFilter, const unsigned int sizeFilter)
{
  //get x and y indices corresponding to current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  //make sure that (xVal, yVal) is within image bounds
  if (withinImageBoundsFilter(xVal, yVal, widthImages, heightImages)) {
    filterImageVerticalProcessPixel<T>(xVal, yVal, imagePixelsToFilter, filteredImagePixels,
      widthImages, heightImages, imageFilter, sizeFilter);
  }
}
