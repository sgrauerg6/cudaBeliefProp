/*
 * SmoothImageCUDA.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCUDA.h"
#include "kernelFilter.cu"

//for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
void SmoothImageCUDA::operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image)
{
  // setup execution parameters
  const auto kernelTBlockDims = this->parallel_params_.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kBlurImages), 0});
  const dim3 threads{kernelTBlockDims[0], kernelTBlockDims[1]};
  const dim3 grid{(unsigned int)(ceil((float)in_image.Width() / (float)threads.x)),
                  (unsigned int)(ceil((float)in_image.Height() / (float)threads.y))};

  //if sigma < kMinSigmaValSmooth, don't smooth image...just convert the input image
  //of unsigned ints to an output image of float values
  if (sigma < kMinSigmaValSmooth)
  {
    //declare and allocate the space for the input unsigned int image pixels and the output float image pixels
    unsigned int* originalImageDevice;
    cudaMalloc((void**) &originalImageDevice, (in_image.Width()*in_image.Height()*sizeof(unsigned int)));

    //load image to the device and convert the pixel values to floats stored in smoothed_image
    cudaMemcpy(originalImageDevice, in_image.PointerToPixelsStart(), (in_image.Width()*in_image.Height()*sizeof(unsigned int)),
      cudaMemcpyHostToDevice);

    //call kernel to convert input unsigned int pixels to output float pixels on the device
    beliefpropCUDA::convertUnsignedIntImageToFloat <<< grid, threads >>> (originalImageDevice, smoothed_image,
      in_image.Width(), in_image.Height());
    cudaDeviceSynchronize();

    //free the device memory used to store original images
    cudaFree(originalImageDevice);
  }
  else
  {
    //apply a Guassian filter to the images
    //retrieve output filter (float array in unique_ptr) and size
    const auto filterWSize = this->MakeFilter(sigma);

    //copy the image filter to the GPU
    float* filterDevice;
    cudaMalloc((void**)&filterDevice, filterWSize.second*sizeof(float));
    cudaMemcpy(filterDevice, filterWSize.first.get(), filterWSize.second*sizeof(float), cudaMemcpyHostToDevice);

    //allocate the GPU global memory for the original, intermediate (when the image has been filtered horizontally but not vertically), and final image
    unsigned int* originalImageDevice;
    float* intermediateImageDevice;

    //it is possible to use the same storage for the original and final images...
    cudaMalloc((void**)&originalImageDevice, (in_image.Width()*in_image.Height()*sizeof(unsigned int)));
    cudaMalloc((void**)&intermediateImageDevice, (in_image.Width()*in_image.Height()*sizeof(float)));

    //copy image to GPU memory
    cudaMemcpy(originalImageDevice, in_image.PointerToPixelsStart(), in_image.Width()*in_image.Height()*sizeof(unsigned int),
      cudaMemcpyHostToDevice);

    //first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
    beliefpropCUDA::filterImageAcross<unsigned int> <<< grid, threads >>> (originalImageDevice, intermediateImageDevice,
      in_image.Width(), in_image.Height(), filterDevice, filterWSize.second);
    cudaDeviceSynchronize();

    //now use the vertical filter to complete the smoothing of image on the device
    beliefpropCUDA::filterImageVertical<float> <<< grid, threads >>> (intermediateImageDevice, smoothed_image,
      in_image.Width(), in_image.Height(), filterDevice, filterWSize.second);
    cudaDeviceSynchronize();

    //free the device memory used to store the images
    cudaFree(originalImageDevice);
    cudaFree(intermediateImageDevice);
    cudaFree(filterDevice);
  }
}