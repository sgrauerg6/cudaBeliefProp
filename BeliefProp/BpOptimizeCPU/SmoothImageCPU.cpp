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
    float_image_pixels[y_val * width_images + x_val] = 1.0f * uint_image_pixels[y_val * width_images + x_val];
  }
}
