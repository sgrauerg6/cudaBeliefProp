/*
 * SmoothImageCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCPU.h"

SmoothImageCPU::SmoothImageCPU() {
}

SmoothImageCPU::~SmoothImageCPU() {
}


//function to use the CPU-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
void SmoothImageCPU::operator()(const BpImage<unsigned int>& inImage, float sigmaVal, float* smoothedImage)
{
	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH) {
		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloatCPU(inImage.getPointerToPixelsStart(), smoothedImage, inImage.getWidth(),
				inImage.getHeight());
	}
	//otherwise apply a Guassian filter to the images
	else
	{
		//retrieve output filter (float array in unique_ptr) and size
		auto outFilterAndSize = makeFilter(sigmaVal);
		auto filter = std::move(outFilterAndSize.first);
		unsigned int sizeFilter = outFilterAndSize.second;

		//space for intermediate image (when the image has been filtered horizontally but not vertically)
		std::unique_ptr<float[]> intermediateImage = std::make_unique<float[]>(inImage.getWidth() * inImage.getHeight());

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterImageAcrossCPU<unsigned int>(inImage.getPointerToPixelsStart(), &(intermediateImage[0]), inImage.getWidth(),
				inImage.getHeight(), &(filter[0]), sizeFilter);

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterImageVerticalCPU<float>(&(intermediateImage[0]), smoothedImage,
				inImage.getWidth(), inImage.getHeight(), &(filter[0]), sizeFilter);
	}
}

