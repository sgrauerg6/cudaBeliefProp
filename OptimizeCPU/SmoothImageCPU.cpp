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

//checks if the current point is within the image bounds
bool withinImageBoundsFilterCPU(int xVal, int yVal, int width, int height) {
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

//kernal to convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//output filtered image stored in floatImagePixels
void convertUnsignedIntImageToFloatCPU(
		unsigned int* imagePixelsUnsignedIntToFilter, float* floatImagePixels,
		int widthImages, int heightImages) {
#pragma omp parallel for
	for (int val = 0; val < widthImages * heightImages; val++) {
		int yVal = val / widthImages;
		int xVal = val % widthImages;
		{
			//retrieve the float-value of the unsigned int pixel value at the current location
			float floatPixelVal = 1.0f
					* imagePixelsUnsignedIntToFilter[yVal * widthImages + xVal];

			floatImagePixels[yVal * widthImages + xVal] = floatPixelVal;
		}
	}
}

//kernal to apply a horizontal filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
void filterFloatImageAcrossCPU(float* imagePixelsFloatToFilter,
		float* filteredImagePixels, int widthImages, int heightImages,
		float* imageFilter, int sizeFilter) {
#pragma omp parallel for
	for (int val = 0; val < widthImages * heightImages; val++) {
		int yVal = val / widthImages;
		int xVal = val % widthImages;
		{
			float filteredPixelVal = imageFilter[0]
					* imagePixelsFloatToFilter[yVal * widthImages + xVal];

			for (int i = 1; i < sizeFilter; i++) {
				filteredPixelVal += imageFilter[i]
						* (imagePixelsFloatToFilter[yVal * widthImages
								+ std::max(xVal - i, 0)]
								+ imagePixelsFloatToFilter[yVal * widthImages
										+ std::min(xVal + i, widthImages - 1)]);
			}

			filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
		}
	}
}

//kernal to apply a vertical filter on each pixel of the image in parallel
//input image stored in texture imagePixelsFloatToFilterTexture
//output filtered image stored in filteredImagePixels
void filterFloatImageVerticalCPU(float* imagePixelsFloatToFilter,
		float* filteredImagePixels, int widthImages, int heightImages,
		float* imageFilter, int sizeFilter) {
#pragma omp parallel for
	for (int val = 0; val < widthImages * heightImages; val++) {
		int yVal = val / widthImages;
		int xVal = val % widthImages;
		{
			float filteredPixelVal = imageFilter[0]
					* (imagePixelsFloatToFilter[yVal * widthImages + xVal]);

			for (int i = 1; i < sizeFilter; i++) {
				filteredPixelVal +=
						imageFilter[i]
								* (imagePixelsFloatToFilter[std::max(yVal - i,
										0) * widthImages + xVal]
										+ imagePixelsFloatToFilter[std::min(
												yVal + i, heightImages - 1)
												* widthImages + xVal]);
			}

			filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
		}
	}
}

//kernal to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
void filterUnsignedIntImageAcrossCPU(
		unsigned int* imagePixelsUnsignedIntToFilter,
		float* filteredImagePixels, int widthImages, int heightImages,
		float* imageFilter, int sizeFilter) {
#pragma omp parallel for
	for (int val = 0; val < widthImages * heightImages; val++) {
		int yVal = val / widthImages;
		int xVal = val % widthImages;
		{
			float filteredPixelVal = imageFilter[0]
					* ((float) imagePixelsUnsignedIntToFilter[yVal * widthImages
							+ xVal]);

			for (int i = 1; i < sizeFilter; i++) {
				filteredPixelVal +=
						imageFilter[i]
								* (((float) imagePixelsUnsignedIntToFilter[yVal
										* widthImages + std::max(xVal - i, 0)])
										+ ((float) imagePixelsUnsignedIntToFilter[yVal
												* widthImages
												+ std::min(xVal + i,
														widthImages - 1)]));
			}

			filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
		}
	}
}

//kernal to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
void filterUnsignedIntImageVerticalCPU(
		unsigned int* imagePixelsUnsignedIntToFilter,
		float* filteredImagePixels, int widthImages, int heightImages,
		float* imageFilter, int sizeFilter) {
#pragma omp parallel for
	for (int val = 0; val < widthImages * heightImages; val++) {
		int yVal = val / widthImages;
		int xVal = val % widthImages;
		{
			float filteredPixelVal = imageFilter[0]
					* ((float) imagePixelsUnsignedIntToFilter[yVal * widthImages
							+ xVal]);

			for (int i = 1; i < sizeFilter; i++) {
				filteredPixelVal +=
						imageFilter[i]
								* ((float) (imagePixelsUnsignedIntToFilter[std::max(
										yVal - i, 0) * widthImages + xVal])
										+ ((float) imagePixelsUnsignedIntToFilter[std::min(
												yVal + i, heightImages - 1)
												* widthImages + xVal]));
			}

			filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
		}
	}
}

//function to use the CPU-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
void SmoothImageCPU::operator()(unsigned int* inImage, unsigned int widthImages,
		unsigned int heightImages, float sigmaVal, float* smoothedImage) {
	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH) {
		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloatCPU(inImage, smoothedImage, widthImages,
				heightImages);
	}
	//otherwise apply a Guassian filter to the images
	else {
		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//space for intermediate image (when the image has been filtered horizontally but not vertically)
		float* intermediateImage = new float[widthImages * heightImages];

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcrossCPU(inImage, intermediateImage, widthImages,
				heightImages, filter, sizeFilter);

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVerticalCPU(intermediateImage, smoothedImage,
				widthImages, heightImages, filter, sizeFilter);

		delete[] intermediateImage;
		delete[] filter;
	}
}
