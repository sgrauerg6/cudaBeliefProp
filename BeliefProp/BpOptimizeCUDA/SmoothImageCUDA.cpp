/*
 * SmoothImageCUDA.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCUDA.h"
#include "kernelFilter.cu"

//for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
void SmoothImageCUDA::operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image) const
{
  // setup execution parameters
  const auto kernel_thread_block_dims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0});
  const dim3 threads{kernel_thread_block_dims[0], kernel_thread_block_dims[1]};
  const dim3 grid{(unsigned int)(ceil((float)in_image.Width() / (float)threads.x)),
                  (unsigned int)(ceil((float)in_image.Height() / (float)threads.y))};

  //if sigma < kMinSigmaValSmooth, don't smooth image...just convert the input image
  //of unsigned ints to an output image of float values
  if (sigma < kMinSigmaValSmooth)
  {
    //declare and allocate the space for the input unsigned int image pixels and the output float image pixels
    unsigned int* original_image_device;
    cudaMalloc((void**) &original_image_device, (in_image.Width()*in_image.Height()*sizeof(unsigned int)));

    //load image to the device and convert the pixel values to floats stored in smoothed_image
    cudaMemcpy(original_image_device, in_image.PointerToPixelsStart(), (in_image.Width()*in_image.Height()*sizeof(unsigned int)),
      cudaMemcpyHostToDevice);

    //call kernel to convert input unsigned int pixels to output float pixels on the device
    beliefpropCUDA::convertUnsignedIntImageToFloat <<< grid, threads >>> (original_image_device, smoothed_image,
      in_image.Width(), in_image.Height());
    cudaDeviceSynchronize();

    //free the device memory used to store original images
    cudaFree(original_image_device);
  }
  else
  {
    //apply a Guassian filter to the images
    //retrieve output filter (float array in unique_ptr) and size
    const auto filter_w_size = this->MakeFilter(sigma);

    //copy the image filter to the GPU
    float* filter_device;
    cudaMalloc((void**)&filter_device, filter_w_size.second*sizeof(float));
    cudaMemcpy(filter_device, filter_w_size.first.data(), filter_w_size.second*sizeof(float), cudaMemcpyHostToDevice);

    //allocate the GPU global memory for the original, intermediate (when the image has been filtered horizontally but not vertically), and final image
    unsigned int* original_image_device;
    float* intermediate_image_device;

    //it is possible to use the same storage for the original and final images...
    cudaMalloc((void**)&original_image_device, (in_image.Width()*in_image.Height()*sizeof(unsigned int)));
    cudaMalloc((void**)&intermediate_image_device, (in_image.Width()*in_image.Height()*sizeof(float)));

    //copy image to GPU memory
    cudaMemcpy(original_image_device, in_image.PointerToPixelsStart(), in_image.Width()*in_image.Height()*sizeof(unsigned int),
      cudaMemcpyHostToDevice);

    //first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
    beliefpropCUDA::FilterImageAcross<unsigned int> <<< grid, threads >>> (original_image_device, intermediate_image_device,
      in_image.Width(), in_image.Height(), filter_device, filter_w_size.second);
    cudaDeviceSynchronize();

    //now use the vertical filter to complete the smoothing of image on the device
    beliefpropCUDA::FilterImageVertical<float> <<< grid, threads >>> (intermediate_image_device, smoothed_image,
      in_image.Width(), in_image.Height(), filter_device, filter_w_size.second);
    cudaDeviceSynchronize();

    //free the device memory used to store the images
    cudaFree(original_image_device);
    cudaFree(intermediate_image_device);
    cudaFree(filter_device);
  }
}