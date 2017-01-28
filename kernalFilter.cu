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

//checks if the current point is within the image bounds
__device__ bool withinImageBoundsFilter(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}


//kernal to convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//output filtered image stored in floatImagePixels
__global__ void convertUnsignedIntImageToFloat(float* floatImagePixels)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{
		//retrieve the float-value of the unsigned int pixel value at the current location
		float floatPixelVal = 1.0f*tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, yVal*widthImageConstFilt + xVal);

		floatImagePixels[yVal*widthImageConstFilt + xVal] = floatPixelVal;
	}
}


//kernal to apply a horizontal filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
__global__ void filterFloatImageAcross(float* filteredImagePixels)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{

		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + xVal) ;


		for (int i = 1; i < sizeFilterConst; i++) 
		{
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + max(xVal-i, 0)) 
				+ tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + min(xVal+i, widthImageConstFilt-1))); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}


//kernal to apply a vertical filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
__global__ void filterFloatImageVertical(float* filteredImagePixels)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{

		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsFloatToFilterTexture, yVal*widthImageConstFilt + xVal);


		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsFloatToFilterTexture, max(yVal-i, 0)*widthImageConstFilt + xVal) 
				+ tex1Dfetch(imagePixelsFloatToFilterTexture, min(yVal+i, heightImageConstFilt-1)*widthImageConstFilt + xVal)); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}

//kernal to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
__global__ void filterUnsignedIntImageAcross(float* filteredImagePixels)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{

		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, yVal*widthImageConstFilt + xVal) ;


		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, yVal*widthImageConstFilt + max(xVal-i, 0)) 
				+ tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, yVal*widthImageConstFilt + min(xVal+i, widthImageConstFilt-1))); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}


//kernal to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
__global__ void filterUnsignedIntImageVertical(float* filteredImagePixels)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int xVal = bx * BLOCK_SIZE_WIDTH_FILTER_IMAGES + tx;
	int yVal = by * BLOCK_SIZE_HEIGHT_FILTER_IMAGES + ty;

	//make sure that (xVal, yVal) is within image bounds
	if (withinImageBoundsFilter(xVal, yVal, widthImageConstFilt, heightImageConstFilt))
	{

		float filteredPixelVal = imageFilterConst[0]*tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, yVal*widthImageConstFilt + xVal);


		for (int i = 1; i < sizeFilterConst; i++) {
			filteredPixelVal += imageFilterConst[i] * (tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, max(yVal-i, 0)*widthImageConstFilt + xVal) 
				+ tex1Dfetch(imagePixelsUnsignedIntToFilterTexture, min(yVal+i, heightImageConstFilt-1)*widthImageConstFilt + xVal)); 
		}

		filteredImagePixels[yVal*widthImageConstFilt + xVal] = filteredPixelVal;
	}
}
