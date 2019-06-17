/*
 * SmoothImageCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCPU.h"

SmoothImageCPU::SmoothImageCPU()
{
}

SmoothImageCPU::~SmoothImageCPU()
{
}

//function to use the CPU-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
void SmoothImageCPU::operator()(unsigned int* inImage, int widthImages,
		int heightImages, float sigmaVal, float* imageSmoothed)
{
	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH) {
		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloatCPU(originalImageDevice,
				image1SmoothedDevice, widthImages, heightImages);
	}
	//otherwise apply a Guassian filter to the images
	else {
		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//allocate the GPU global memory for the original, intermediate (when the image has been filtered horizontally but not vertically), and final image
		unsigned int* originalImageDevice;
		float* intermediateImageDevice = new float[widthImages * heightImages];

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcrossCPU(originalImageDevice,
				intermediateImageDevice, widthImages, heightImages, filter,
				sizeFilter);

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVerticalCPU(intermediateImageDevice,
				image1SmoothedDevice, widthImages, heightImages, filter,
				sizeFilter);
	}
}

