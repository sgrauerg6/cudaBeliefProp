/*
 * SharedSmoothImageFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDSMOOTHIMAGEFUNCTS_H_
#define SHAREDSMOOTHIMAGEFUNCTS_H_

#include "RunSettingsEval/RunTypeConstraints.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "RunImp/RunImpGenFuncts.h"

namespace beliefprop {

//kernel to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template <BpImData_t T>
ARCHITECTURE_ADDITION inline void filterImageAcrossProcessPixel(const unsigned int xVal, const unsigned int yVal,
  T* imagePixelsToFilter, float* filteredImagePixels, const unsigned int widthImages, const unsigned int heightImages,
  float* imageFilter, const unsigned int sizeFilter)
{
  float filteredPixelVal = imageFilter[0] * ((float)imagePixelsToFilter[yVal*widthImages + xVal]);

  for (unsigned int i = 1; i < sizeFilter; i++) {
    filteredPixelVal += imageFilter[i] * (((float)imagePixelsToFilter[yVal*widthImages + (unsigned int)GenProcessingFuncts::getMax((int)xVal - (int)i, 0)]) +
      ((float)imagePixelsToFilter[yVal*widthImages + GenProcessingFuncts::getMin(xVal + i, widthImages - 1)]));
  }

  filteredImagePixels[yVal*widthImages + xVal] = filteredPixelVal;
}

//kernel to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template <BpImData_t T>
ARCHITECTURE_ADDITION inline void filterImageVerticalProcessPixel(const unsigned int xVal, const unsigned int yVal,
  T* imagePixelsToFilter, float* filteredImagePixels, const unsigned int widthImages, const unsigned int heightImages,
  float* imageFilter, const unsigned int sizeFilter)
{
  float filteredPixelVal = imageFilter[0] * ((float)imagePixelsToFilter[yVal*widthImages + xVal]);

  for (unsigned int i = 1; i < sizeFilter; i++) {
    filteredPixelVal += imageFilter[i] * ((float) (imagePixelsToFilter[(unsigned int)GenProcessingFuncts::getMax((int)yVal - (int)i, 0) * widthImages + xVal]) +
      ((float)imagePixelsToFilter[GenProcessingFuncts::getMin(yVal + i, heightImages - 1) * widthImages + xVal]));
  }

  filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
}

}

#endif /* SHAREDSMOOTHIMAGEFUNCTS_H_ */
