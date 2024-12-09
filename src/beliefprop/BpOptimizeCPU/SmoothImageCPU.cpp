/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file SmoothImageCPU.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "BpSharedFuncts/SharedSmoothImageFuncts.h"
#include "SmoothImageCPU.h"

//function to use the CPU-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigma
void SmoothImageCPU::operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image) const
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
    const auto filter_w_size = this->MakeFilter(sigma);

    //space for intermediate image (when the image has been filtered horizontally but not vertically)
    std::unique_ptr<float[]> intermediateImage = std::make_unique<float[]>(in_image.Width() * in_image.Height());

     //first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
    FilterImageAcrossCPU<unsigned int>(in_image.PointerToPixelsStart(), intermediateImage.get(), in_image.Width(),
      in_image.Height(), filter_w_size.first.data(), filter_w_size.second, this->parallel_params_);

    //now use the vertical filter to complete the smoothing of image 1 on the device
    FilterImageVerticalCPU<float>(intermediateImage.get(), smoothed_image,
      in_image.Width(), in_image.Height(), filter_w_size.first.data(), filter_w_size.second, this->parallel_params_);
  }
}

//convert the unsigned int pixels to float pixels in an image when
//smoothing is not desired but the pixels need to be converted to floats
//the input image is stored as unsigned ints in the texture uint_image_pixels
//output filtered image stored in float_image_pixels
void SmoothImageCPU::ConvertUnsignedIntImageToFloatCPU(
    const unsigned int* uint_image_pixels, float* float_image_pixels,
    unsigned int width_images, unsigned int height_images,
    const ParallelParams& opt_cpu_params) const
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{
    (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
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
    float_image_pixels[y_val * width_images + x_val] = 1.0f * uint_image_pixels[y_val * width_images + x_val];
  }
}


//apply a horizontal filter on each pixel of the image in parallel
template<BpImData_t U>
void SmoothImageCPU::FilterImageAcrossCPU(
  const U* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params) const
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{
    (int)opt_cpu_params.OptParamsForKernel(
      {static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
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
    beliefprop::FilterImageAcrossProcessPixel<U>(
      x_val, y_val, image_to_filter, filtered_image,
      width_images, height_images, image_filter, size_filter);
  }
}

//apply a vertical filter on each pixel of the image in parallel
template<BpImData_t U>
void SmoothImageCPU::FilterImageVerticalCPU(
  const U* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params) const
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{
    (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0})[0]};
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
    beliefprop::FilterImageVerticalProcessPixel<U>(
      x_val, y_val, image_to_filter, filtered_image,
      width_images, height_images, image_filter, size_filter);
  }
}

//explicit instantiations of template member functions
//which take float or unsigned int as templated type
//TODO: seems to compile without explicit instantiations,
//so not clear if needed or not
//explicit instantiations for FilterImageAcrossCPU member
//function
template void SmoothImageCPU::FilterImageAcrossCPU<float>(
  const float* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params) const;
template void SmoothImageCPU::FilterImageAcrossCPU<unsigned int>(
  const unsigned int* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params) const;

//explicit instantiations for FilterImageVerticalCPU member
//function
template void SmoothImageCPU::FilterImageVerticalCPU<float>(
  const float* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params) const;
template void SmoothImageCPU::FilterImageVerticalCPU<unsigned int>(
  const unsigned int* image_to_filter, float* filtered_image,
  unsigned int width_images, unsigned int height_images,
  const float* image_filter, unsigned int size_filter,
  const ParallelParams& opt_cpu_params) const;