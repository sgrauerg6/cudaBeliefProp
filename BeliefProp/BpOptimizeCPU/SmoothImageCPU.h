/*
 * SmoothImageCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECPU_H_
#define SMOOTHIMAGECPU_H_

#include <utility>
#include <memory>
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpSharedFuncts/SharedSmoothImageFuncts.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunImp/BpParallelParams.h"

class SmoothImageCPU final : public SmoothImage {
public:
  SmoothImageCPU(const ParallelParams& optCPUParams) : SmoothImage(optCPUParams) {}

  //function to use the CPU-image filter to apply a guassian filter to the a single images
  //input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
  //output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigma
  void operator()(const BpImage<unsigned int>& in_image, const float sigma, float* smoothed_image) override;

private:  
  //convert the unsigned int pixels to float pixels in an image when
  //smoothing is not desired but the pixels need to be converted to floats
  //the input image is stored as unsigned ints in the texture imagePixelsUnsignedInt
  //output filtered image stored in floatImagePixels
  void ConvertUnsignedIntImageToFloatCPU(unsigned int* imagePixelsUnsignedInt, float* floatImagePixels,
    unsigned int widthImages, unsigned int heightImages,
    const ParallelParams& optCPUParams);

  //apply a horizontal filter on each pixel of the image in parallel
  template<BpImData_t U>
  void FilterImageAcrossCPU(U* imagePixelsToFilter, float* filteredImagePixels,
    unsigned int widthImages, unsigned int heightImages,
    float* imageFilter, unsigned int size_filter,
    const ParallelParams& optCPUParams);

  //apply a vertical filter on each pixel of the image in parallel
  template<BpImData_t U>
  void FilterImageVerticalCPU(U* imagePixelsToFilter, float* filteredImagePixels,
    unsigned int widthImages, unsigned int heightImages,
    float* imageFilter, unsigned int size_filter,
    const ParallelParams& optCPUParams);
};

//apply a horizontal filter on each pixel of the image in parallel
template<BpImData_t U>
void SmoothImageCPU::FilterImageAcrossCPU(U* imagePixelsToFilter, float* filteredImagePixels,
  unsigned int widthImages, unsigned int heightImages,
  float* imageFilter, unsigned int size_filter,
  const ParallelParams& optCPUParams)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int numThreadsKernel{(int)optCPUParams.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
  #pragma omp parallel for num_threads(numThreadsKernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < widthImages * heightImages; val++) {
#else
  for (unsigned int val = 0; val < widthImages * heightImages; val++) {
#endif //_WIN32
    const unsigned int yVal = val / widthImages;
    const unsigned int xVal = val % widthImages;
    beliefprop::FilterImageAcrossProcessPixel<U>(xVal, yVal, imagePixelsToFilter, filteredImagePixels,
      widthImages, heightImages, imageFilter, size_filter);
  }
}

//apply a vertical filter on each pixel of the image in parallel
template<BpImData_t U>
void SmoothImageCPU::FilterImageVerticalCPU(U* imagePixelsToFilter, float* filteredImagePixels,
  unsigned int widthImages, unsigned int heightImages,
  float* imageFilter, unsigned int size_filter,
  const ParallelParams& optCPUParams)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int numThreadsKernel{(int)optCPUParams.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
  #pragma omp parallel for num_threads(numThreadsKernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < widthImages * heightImages; val++) {
#else
  for (unsigned int val = 0; val < widthImages * heightImages; val++) {
#endif //_WIN32
    const unsigned int yVal = val / widthImages;
    const unsigned int xVal = val % widthImages;
    beliefprop::FilterImageVerticalProcessPixel<U>(xVal, yVal, imagePixelsToFilter,
      filteredImagePixels, widthImages, heightImages, imageFilter, size_filter);
  }
}

#endif /* SMOOTHIMAGECPU_H_ */
