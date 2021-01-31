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

//This kernal is used to filter the image with the given filter in the vertical and horizontal directions


#include "kernalFilterHeader.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include "../ParameterFiles/bpStereoCudaParameters.h"

#define PROCESSING_ON_GPU
#include "../SharedFuncts/SharedSmoothImageFuncts.h"
#undef PROCESSING_ON_GPU

//checks if the current point is within the image bounds
__device__ bool withinImageBoundsFilter(const unsigned int xVal, const unsigned int yVal, const unsigned int width, const unsigned int height)
{
	//xVal and yVal unsigned so no need to compare with zero
	return (/*(xVal >= 0) &&*/ (xVal < width) && /*(yVal >= 0) &&*/ (yVal < height));
}


//kernal to convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//output filtered image stored in floatImagePixels
__global__ void convertUnsignedIntImageToFloat(
		unsigned int* imagePixelsUnsignedIntToFilter, float* floatImagePixels,
		const unsigned int widthImages, const unsigned int heightImages)
{
	// Block index
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	// Thread index
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const unsigned int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	const unsigned int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImages, heightImages))
	{
		//retrieve the float-value of the unsigned int pixel value at the current location
		const float floatPixelVal = 1.0f * imagePixelsUnsignedIntToFilter[yVal*widthImages + xVal];

		floatImagePixels[yVal*widthImages + xVal] = floatPixelVal;
	}
}


//kernal to apply a horizontal filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
template<typename T>
__global__ void filterImageAcross(T* imagePixelsToFilter, float* filteredImagePixels,
		const unsigned int widthImages, const unsigned int heightImages,
		float* imageFilter, const unsigned int sizeFilter)
{
	// Block index
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	// Thread index
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const unsigned int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	const unsigned int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImages, heightImages))
	{
		filterImageAcrossProcessPixel<T>(xVal, yVal,
				imagePixelsToFilter,
				filteredImagePixels, widthImages, heightImages,
				imageFilter, sizeFilter);
	}
}


//kernal to apply a vertical filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
template<typename T>
__global__ void filterImageVertical(T* imagePixelsToFilter, float* filteredImagePixels,
		const unsigned int widthImages, const unsigned int heightImages, float* imageFilter, const unsigned int sizeFilter)
{
	// Block index
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	// Thread index
	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const unsigned int xVal = bx * bp_cuda_params::BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	const unsigned int yVal = by * bp_cuda_params::BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImages, heightImages))
	{
		filterImageVerticalProcessPixel<T>(xVal, yVal,
				imagePixelsToFilter,
				filteredImagePixels, widthImages, heightImages,
				imageFilter, sizeFilter);
	}
}
