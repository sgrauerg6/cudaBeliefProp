/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//Defines the functions used to smooth the input images with a gaussian filter of SIGMA_BP (see bpCudaParameters.cuh) implemented in CUDA

#include "smoothImageHostHeader.cuh"


//functions relating to smoothing the images before running BP


/* normalize mask so it integrates to one */
__host__ void normalizeFilter(float*& filter, int sizeFilter) 
{
	float sum = 0;
	for (int i = 1; i < sizeFilter; i++) 
	{
		sum += fabs(filter[i]);
	}
	sum = 2*sum + fabs(filter[0]);
	for (int i = 0; i < sizeFilter; i++) 
	{
		filter[i] /= sum;
	}
}

//this function creates a Gaussian filter given a sigma value
__host__ float* makeFilter(float sigma, int& sizeFilter)
{
	sigma = max(sigma, 0.01f);
	sizeFilter = (int)ceil(sigma * WIDTH_SIGMA_1) + 1;
	float* mask = new float[sizeFilter];
	for (int i = 0; i < sizeFilter; i++) 
	{
		mask[i] = exp(-0.5*((i/sigma) * (i/sigma)));
	}
	normalizeFilter(mask, sizeFilter);

	return mask;
}


//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothSingleImageInputHostOutputDeviceCUDA(unsigned int*& image1InHost, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice)
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

		(cudaMalloc((void**) &originalImageDevice, (widthImages*heightImages*sizeof(unsigned int))));

		//load image 1 to the device and convert the pixel values to floats stored in image1SmoothedDevice

		//loads image 1 to the device
		(cudaMemcpy(originalImageDevice, image1InHost, (widthImages*heightImages*sizeof(unsigned int)),
								cudaMemcpyHostToDevice));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(originalImageDevice, image1SmoothedDevice, widthImages, heightImages);

		( cudaDeviceSynchronize() );

		//free the device memory used to store original images
		(cudaFree(originalImageDevice));
	}

	//otherwise apply a Guassian filter to the images
	else
	{

		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter to the GPU
		float* filterDevice;
		(cudaMalloc((void**) &filterDevice, (sizeFilter*sizeof(float))));
		cudaMemcpy(filterDevice, filter, (sizeFilter*sizeof(float)), cudaMemcpyHostToDevice);

		//allocate the GPU global memory for the original, intermediate (when the image has been filtered horizontally but not vertically), and final image
		unsigned int* originalImageDevice;
		float* intermediateImageDevice; 

		//it is possible to use the same storage for the original and final images...
		(cudaMalloc((void**) &originalImageDevice, (widthImages*heightImages*sizeof(unsigned int))));
		(cudaMalloc((void**) &intermediateImageDevice, (widthImages*heightImages*sizeof(float))));

		//first smooth the image 1, so copy image 1 to GPU memory

		//load image 1 to the device
		(cudaMemcpy(originalImageDevice, image1InHost, (widthImages*heightImages*sizeof(unsigned int)),
									cudaMemcpyHostToDevice));

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (originalImageDevice, intermediateImageDevice, widthImages, heightImages, filterDevice, sizeFilter);

		( cudaDeviceSynchronize() );

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (intermediateImageDevice, image1SmoothedDevice, widthImages, heightImages, filterDevice, sizeFilter);

		( cudaDeviceSynchronize() );

		//free the device memory used to store the images
		(cudaFree(originalImageDevice));
		(cudaFree(intermediateImageDevice));
		cudaFree(filterDevice);
	}
}
