/*
 * SmoothImageCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCPU.h"

//function to use the CPU-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigma
void SmoothImageCPU::operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image)
{
  //if sigma < kMinSigmaValSmooth, then don't smooth image...just convert the input image
  //of unsigned ints to an output image of float values
  if (sigma < kMinSigmaValSmooth) {
    //call kernel to convert input unsigned int pixels to output float pixels on the device
    ConvertUnsignedIntImageToFloatCPU(in_image.PointerToPixelsStart(), smoothed_image,
      in_image.Width(), in_image.Height(), this->parallel_params_);
  }
  //otherwise apply a Guassian filter to the images
  else
  {
    //retrieve output filter (float array in unique_ptr) and size
    const auto filterWSize = this->MakeFilter(sigma);

    //space for intermediate image (when the image has been filtered horizontally but not vertically)
    std::unique_ptr<float[]> intermediateImage = std::make_unique<float[]>(in_image.Width() * in_image.Height());

     //first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
    FilterImageAcrossCPU<unsigned int>(in_image.PointerToPixelsStart(), intermediateImage.get(), in_image.Width(),
      in_image.Height(), filterWSize.first.get(), filterWSize.second, this->parallel_params_);

    //now use the vertical filter to complete the smoothing of image 1 on the device
    FilterImageVerticalCPU<float>(intermediateImage.get(), smoothed_image,
      in_image.Width(), in_image.Height(), filterWSize.first.get(), filterWSize.second, this->parallel_params_);
  }
}

//convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture imagePixelsUnsignedInt
//output filtered image stored in floatImagePixels
void SmoothImageCPU::ConvertUnsignedIntImageToFloatCPU(unsigned int* imagePixelsUnsignedInt, float* floatImagePixels,
    unsigned int widthImages, unsigned int heightImages,
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
    floatImagePixels[yVal * widthImages + xVal] = 1.0f * imagePixelsUnsignedInt[yVal * widthImages + xVal];
  }
}
