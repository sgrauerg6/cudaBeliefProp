/*
 * SmoothImageCUDA.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCUDA.h"
#include "kernelFilter.cu"

//for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
void SmoothImageCUDA::operator()(const BpImage<unsigned int>& inImage, const float sigmaVal, float* smoothedImage)
{
  // setup execution parameters
  const dim3 threads{cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][0],
                     cudaParams_.parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][1]};
  const dim3 grid{(unsigned int)(ceil((float)inImage.getWidth() / (float)threads.x)),
                  (unsigned int)(ceil((float)inImage.getHeight() / (float)threads.y))};

  //if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
  //of unsigned ints to an output image of float values
  if (sigmaVal < MIN_SIGMA_VAL_SMOOTH)
  {
    //declare and allocate the space for the input unsigned int image pixels and the output float image pixels
    unsigned int* originalImageDevice;
    cudaMalloc((void**) &originalImageDevice, (inImage.getWidth()*inImage.getHeight()*sizeof(unsigned int)));

    //load image to the device and convert the pixel values to floats stored in smoothedImage
    cudaMemcpy(originalImageDevice, inImage.getPointerToPixelsStart(), (inImage.getWidth()*inImage.getHeight()*sizeof(unsigned int)),
      cudaMemcpyHostToDevice);

    //call kernel to convert input unsigned int pixels to output float pixels on the device
    beliefpropCUDA::convertUnsignedIntImageToFloat <<< grid, threads >>> (originalImageDevice, smoothedImage, inImage.getWidth(), inImage.getHeight());
    cudaDeviceSynchronize();

    //free the device memory used to store original images
    cudaFree(originalImageDevice);
  }
  else
  {
    //apply a Guassian filter to the images
    //retrieve output filter (float array in unique_ptr) and size
    auto outFilterAndSize = this->makeFilter(sigmaVal);
    auto filter = std::move(outFilterAndSize.first);
    unsigned int sizeFilter = outFilterAndSize.second;

    //copy the image filter to the GPU
    float* filterDevice;
    cudaMalloc((void**)&filterDevice, (sizeFilter*sizeof(float)));
    cudaMemcpy(filterDevice, &(filter[0]), (sizeFilter*sizeof(float)), cudaMemcpyHostToDevice);

    //allocate the GPU global memory for the original, intermediate (when the image has been filtered horizontally but not vertically), and final image
    unsigned int* originalImageDevice;
    float* intermediateImageDevice;

    //it is possible to use the same storage for the original and final images...
    cudaMalloc((void**)&originalImageDevice, (inImage.getWidth()*inImage.getHeight()*sizeof(unsigned int)));
    cudaMalloc((void**)&intermediateImageDevice, (inImage.getWidth()*inImage.getHeight()*sizeof(float)));

    //copy image to GPU memory
    cudaMemcpy(originalImageDevice, inImage.getPointerToPixelsStart(), (inImage.getWidth()*inImage.getHeight()*sizeof(unsigned int)),
      cudaMemcpyHostToDevice);

    //first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
    beliefpropCUDA::filterImageAcross<unsigned int> <<< grid, threads >>> (originalImageDevice, intermediateImageDevice, inImage.getWidth(), inImage.getHeight(),
      filterDevice, sizeFilter);
    cudaDeviceSynchronize();

    //now use the vertical filter to complete the smoothing of image on the device
    beliefpropCUDA::filterImageVertical<float> <<< grid, threads >>> (intermediateImageDevice, smoothedImage, inImage.getWidth(), inImage.getHeight(),
        filterDevice, sizeFilter);
    cudaDeviceSynchronize();

    //free the device memory used to store the images
    cudaFree(originalImageDevice);
    cudaFree(intermediateImageDevice);
    cudaFree(filterDevice);
  }
}