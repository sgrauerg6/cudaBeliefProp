/*
 * SmoothImageCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECPU_H_
#define SMOOTHIMAGECPU_H_

#include "SmoothImage.h"
#include <algorithm>
#include "../SharedFuncts/SharedSmoothImageFuncts.h"


class SmoothImageCPU : public SmoothImage {

public:
	SmoothImageCPU();
	virtual ~SmoothImageCPU();

	void operator()(unsigned int* inImage, unsigned int widthImages,
			unsigned int heightImages, float sigmaVal, float* smoothedImage);

private:

	//convert the unsigned int pixels to float pixels in an image when
	//smoothing is not desired but the pixels need to be converted to floats
	//the input image is stored as unsigned ints in the texture imagePixelsUnsignedInt
	//output filtered image stored in floatImagePixels
	void convertUnsignedIntImageToFloatCPU(
			unsigned int* imagePixelsUnsignedInt,
			float* floatImagePixels, int widthImages, int heightImages)
	{
		#pragma omp parallel for
		for (int val = 0; val < widthImages * heightImages; val++) {
			int yVal = val / widthImages;
			int xVal = val % widthImages;
			{
				floatImagePixels[yVal * widthImages + xVal] = 1.0f
						* imagePixelsUnsignedInt[yVal * widthImages
								+ xVal];
			}
		}
	}

	//apply a horizontal filter on each pixel of the image in parallel
	template<typename T>
	void filterImageAcrossCPU(T* imagePixelsToFilter,
			float* filteredImagePixels, int widthImages, int heightImages,
			float* imageFilter, int sizeFilter)
	{
		#pragma omp parallel for
		for (int val = 0; val < widthImages * heightImages; val++) {
			int yVal = val / widthImages;
			int xVal = val % widthImages;
			{
				filterImageAcrossProcessPixel<T>(xVal, yVal,
						imagePixelsToFilter, filteredImagePixels, widthImages,
						heightImages, imageFilter, sizeFilter);
			}
		}
	}

	//apply a vertical filter on each pixel of the image in parallel
	template<typename T>
	void filterImageVerticalCPU(T* imagePixelsToFilter,
			float* filteredImagePixels, int widthImages, int heightImages,
			float* imageFilter, int sizeFilter)
	{
		#pragma omp parallel for
		for (int val = 0; val < widthImages * heightImages; val++) {
			int yVal = val / widthImages;
			int xVal = val % widthImages;
			{
				filterImageVerticalProcessPixel<T>(xVal, yVal,
						imagePixelsToFilter, filteredImagePixels, widthImages,
						heightImages, imageFilter, sizeFilter);
			}
		}
	}

};

#endif /* SMOOTHIMAGECPU_H_ */
