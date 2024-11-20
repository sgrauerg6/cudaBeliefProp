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
//#include "BpRunProcessing/BpStructsAndEnums.h"
#include "BpSharedFuncts/SharedSmoothImageFuncts.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpImageProcessing/BpImageConstraints.h"
#include "BpRunProcessing/BpParallelParams.h"

class SmoothImageCPU final : public SmoothImage {
public:
  SmoothImageCPU(const ParallelParams& opt_cpu_params) : SmoothImage(opt_cpu_params) {}

  //function to use the CPU image filter to apply a guassian filter to the a single images
  //input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
  //output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigma
  void operator()(const BpImage<unsigned int>& in_image, const float sigma, float* smoothed_image) override;

private:
  //convert the unsigned int pixels to float pixels in an image when
  //smoothing is not desired but the pixels need to be converted to floats
  //the input image is stored as unsigned ints in the texture uint_image_pixels
  //output filtered image stored in float_image_pixels
  void ConvertUnsignedIntImageToFloatCPU(
    const unsigned int* uint_image_pixels, float* float_image_pixels,
    unsigned int width_images, unsigned int height_images,
    const ParallelParams& opt_cpu_params);

  //apply a horizontal filter on each pixel of the image in parallel
  template<BpImData_t U>
  void FilterImageAcrossCPU(
    const U* image_to_filter, float* filtered_image,
    unsigned int width_images, unsigned int height_images,
    const float* image_filter, unsigned int size_filter,
    const ParallelParams& opt_cpu_params);

  //apply a vertical filter on each pixel of the image in parallel
  template<BpImData_t U>
  void FilterImageVerticalCPU(
    const U* image_to_filter, float* filtered_image,
    unsigned int width_images, unsigned int height_images,
    const float* image_filter, unsigned int size_filter,
    const ParallelParams& opt_cpu_params);
};

//apply a horizontal filter on each pixel of the image in parallel
template<BpImData_t U>
void SmoothImageCPU::FilterImageAcrossCPU(
  const U* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < width_images * height_images; val++) {
#else
  for (unsigned int val = 0; val < width_images * height_images; val++) {
#endif //_WIN32
    const unsigned int y_val = val / width_images;
    const unsigned int x_val = val % width_images;
    beliefprop::FilterImageAcrossProcessPixel<U>(x_val, y_val, image_to_filter, filtered_image,
      width_images, height_images, image_filter, size_filter);
  }
}

//apply a vertical filter on each pixel of the image in parallel
template<BpImData_t U>
void SmoothImageCPU::FilterImageVerticalCPU(
  const U* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < width_images * height_images; val++) {
#else
  for (unsigned int val = 0; val < width_images * height_images; val++) {
#endif //_WIN32
    const unsigned int y_val = val / width_images;
    const unsigned int x_val = val % width_images;
    beliefprop::FilterImageVerticalProcessPixel<U>(x_val, y_val, image_to_filter,
      filtered_image, width_images, height_images, image_filter, size_filter);
  }
}

#endif /* SMOOTHIMAGECPU_H_ */
