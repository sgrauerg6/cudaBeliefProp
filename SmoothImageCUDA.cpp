/*
 * SmoothImageCUDA.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "SmoothImageCUDA.h"
#include "kernalFilter.cu"

SmoothImageCUDA::SmoothImageCUDA()
{
}

SmoothImageCUDA::~SmoothImageCUDA()
{
}

//for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
void SmoothImageCUDA::operator()(unsigned int* inImage, unsigned int widthImages, unsigned int heightImages, float sigmaVal, float* smoothedImage)
{
	// setup execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_FILTER_IMAGES, BLOCK_SIZE_HEIGHT_FILTER_IMAGES);
	dim3 grid((unsigned int)(ceil((float)widthImages / (float)threads.x)), (unsigned int)(ceil((float)heightImages / (float)threads.y)));

	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH)
	{
		//declare and allocate the space for the input unsigned int image pixels and the output float image pixels
		unsigned int* originalImageDevice;
		cudaMalloc((void**) &originalImageDevice, (widthImages*heightImages*sizeof(unsigned int)));

		//load image to the device and convert the pixel values to floats stored in smoothedImage
		cudaMemcpy(originalImageDevice, inImage, (widthImages*heightImages*sizeof(unsigned int)),
								cudaMemcpyHostToDevice);

		//call kernel to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(inImage, smoothedImage, widthImages, heightImages);

		cudaDeviceSynchronize();

		//free the device memory used to store original images
		cudaFree(originalImageDevice);
	}
	//otherwise apply a Guassian filter to the images
	else
	{
		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter to the GPU
		float* filterDevice;
		cudaMalloc((void**) &filterDevice, (sizeFilter*sizeof(float)));
		cudaMemcpy(filterDevice, filter, (sizeFilter*sizeof(float)), cudaMemcpyHostToDevice);

		//allocate the GPU global memory for the original, intermediate (when the image has been filtered horizontally but not vertically), and final image
		unsigned int* originalImageDevice;
		float* intermediateImageDevice;

		//it is possible to use the same storage for the original and final images...
		cudaMalloc((void**) &originalImageDevice, (widthImages*heightImages*sizeof(unsigned int)));
		cudaMalloc((void**) &intermediateImageDevice, (widthImages*heightImages*sizeof(float)));

		//copy image to GPU memory
		cudaMemcpy(originalImageDevice, inImage, (widthImages*heightImages*sizeof(unsigned int)),
									cudaMemcpyHostToDevice);

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (originalImageDevice, intermediateImageDevice, widthImages, heightImages, filterDevice, sizeFilter);

		cudaDeviceSynchronize();

		//now use the vertical filter to complete the smoothing of image on the device
		filterFloatImageVertical <<< grid, threads >>> (intermediateImageDevice, smoothedImage, widthImages, heightImages, filterDevice, sizeFilter);

		cudaDeviceSynchronize();

		//free the device memory used to store the images
		cudaFree(originalImageDevice);
		cudaFree(intermediateImageDevice);
		cudaFree(filterDevice);
	}
}
