/*
 * SharedSmoothImageFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDSMOOTHIMAGEFUNCTS_H_
#define SHAREDSMOOTHIMAGEFUNCTS_H_

#include "RunSettingsEval/RunTypeConstraints.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "RunImp/RunImpGenFuncts.h"

namespace beliefprop {

//kernel to apply a horizontal filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template <BpImData_t T>
ARCHITECTURE_ADDITION inline void filterImageAcrossProcessPixel(unsigned int xVal, unsigned int yVal,
  T* imagePixelsToFilter, float* filteredImagePixels, unsigned int widthImages, unsigned int heightImages,
  float* imageFilter, unsigned int sizeFilter)
{
  float filteredPixelVal = imageFilter[0] * ((float)imagePixelsToFilter[yVal*widthImages + xVal]);

  for (unsigned int i = 1; i < sizeFilter; i++) {
    filteredPixelVal += imageFilter[i] * (((float)imagePixelsToFilter[yVal*widthImages + (unsigned int)run_imp_util::getMax((int)xVal - (int)i, 0)]) +
      ((float)imagePixelsToFilter[yVal*widthImages + run_imp_util::getMin(xVal + i, widthImages - 1)]));
  }

  filteredImagePixels[yVal*widthImages + xVal] = filteredPixelVal;
}

//kernel to apply a vertical filter on each pixel of the image in parallel
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedIntToFilterTexture
//the output filtered image is returned as an array of floats
template <BpImData_t T>
ARCHITECTURE_ADDITION inline void filterImageVerticalProcessPixel(unsigned int xVal, unsigned int yVal,
  T* imagePixelsToFilter, float* filteredImagePixels, unsigned int widthImages, unsigned int heightImages,
  float* imageFilter, unsigned int sizeFilter)
{
  float filteredPixelVal = imageFilter[0] * ((float)imagePixelsToFilter[yVal*widthImages + xVal]);

  for (unsigned int i = 1; i < sizeFilter; i++) {
    filteredPixelVal += imageFilter[i] * ((float) (imagePixelsToFilter[(unsigned int)run_imp_util::getMax((int)yVal - (int)i, 0) * widthImages + xVal]) +
      ((float)imagePixelsToFilter[run_imp_util::getMin(yVal + i, heightImages - 1) * widthImages + xVal]));
  }

  filteredImagePixels[yVal * widthImages + xVal] = filteredPixelVal;
}

}

#endif /* SHAREDSMOOTHIMAGEFUNCTS_H_ */
