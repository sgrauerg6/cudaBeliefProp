/*
 * SharedSmoothImageFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDSMOOTHIMAGEFUNCTS_H_
#define SHAREDSMOOTHIMAGEFUNCTS_H_

#include "SharedUtilFuncts.h"

//kernal to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template <typename T>
ARCHITECTURE_ADDITION inline void filterImageAcrossProcessPixel(int xVal, int yVal,
		T* imagePixelsToFilter,
		float* filteredImagePixels, int widthImages, int heightImages,
		float* imageFilter, int sizeFilter)
{
	float filteredPixelVal = imageFilter[0]
	* ((float) imagePixelsToFilter[yVal * widthImages
			+ xVal]);

	for (int i = 1; i < sizeFilter; i++) {
		filteredPixelVal +=
		imageFilter[i]
		* (((float) imagePixelsToFilter[yVal
						* widthImages + getMax(xVal - i, 0)])
				+ ((float) imagePixelsToFilter[yVal
						* widthImages
						+ getMin(xVal + i,
								widthImages - 1)]));
	}

	filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
}

//kernal to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template <typename T>
ARCHITECTURE_ADDITION inline void filterImageVerticalProcessPixel(int xVal, int yVal,
		T* imagePixelsToFilter,
		float* filteredImagePixels, int widthImages, int heightImages,
		float* imageFilter, int sizeFilter) {
	float filteredPixelVal =
			imageFilter[0]
					* ((float) imagePixelsToFilter[yVal * widthImages
							+ xVal]);

	for (int i = 1; i < sizeFilter; i++) {
		filteredPixelVal +=
				imageFilter[i]
						* ((float) (imagePixelsToFilter[getMax(
								yVal - i, 0) * widthImages + xVal])
								+ ((float) imagePixelsToFilter[getMin(
										yVal + i, heightImages - 1)
										* widthImages + xVal]));
	}

	filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
}



#endif /* SHAREDSMOOTHIMAGEFUNCTS_H_ */
