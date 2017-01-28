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
		mask[i] = exp(-0.5f*((i/sigma) * (i/sigma)));
	}
	normalizeFilter(mask, sizeFilter);

	return mask;
}

	

//function to use the CUDA-image filter to apply a guassian filter to the two images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothImages(unsigned int*& image1InHost, unsigned int*& image2InHost, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice, float*& image2SmoothedDevice)
{
	// setup execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_FILTER_IMAGES, BLOCK_SIZE_HEIGHT_FILTER_IMAGES);
	dim3 grid((unsigned int)(ceil((float)widthImages / (float)threads.x)), (unsigned int)(ceil((float)heightImages / (float)threads.y)));

	//copy the width and height of the images to the constant memory of the device
	cudaMemcpyToSymbol(widthImageConstFilt, &widthImages, sizeof(int));
	cudaMemcpyToSymbol(heightImageConstFilt, &heightImages, sizeof(int));

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

		//bind the first input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, originalImageDevice, widthImages*heightImages*sizeof(unsigned int));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(image1SmoothedDevice);

		( cudaThreadSynchronize() );

		//now do the same thing with image 2: load to device and convert the pixel values to floats stored in image2SmoothedDevice

		//loads image 2 to the device
		(cudaMemcpy(originalImageDevice, image2InHost, (widthImages*heightImages*sizeof(unsigned int)),
								cudaMemcpyHostToDevice));

		//bind the second input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, originalImageDevice, widthImages*heightImages*sizeof(unsigned int));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(image2SmoothedDevice);

		( cudaThreadSynchronize() );


		//unbind the texture the unsigned int textured that the input images were bound to
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
	

		//free the device memory used to store original images
		(cudaFree(originalImageDevice));
	}

	//otherwise apply a Guassian filter to the images
	else
	{

		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter and the size of the filter to constant memory on the GPU
		cudaMemcpyToSymbol(imageFilterConst, filter, sizeFilter*sizeof(float));
		cudaMemcpyToSymbol(sizeFilterConst, &sizeFilter, sizeof(int));

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

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, originalImageDevice, widthImages*heightImages*sizeof(unsigned int));

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		( cudaThreadSynchronize() );

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (image1SmoothedDevice);

		( cudaThreadSynchronize() );

	

		//then smooth image 2 in the same manner
		(cudaMemcpy(originalImageDevice, image2InHost, (widthImages*heightImages*sizeof(unsigned int)),
									cudaMemcpyHostToDevice) );

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, originalImageDevice, widthImages*heightImages*sizeof(unsigned int));

		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		( cudaThreadSynchronize() );

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (image2SmoothedDevice);

		( cudaThreadSynchronize() );



		//unbind the texture the unsigned int and float textures used for the smoothing
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
		cudaUnbindTexture( imagePixelsFloatToFilterTexture);

		//free the device memory used to store the images
		(cudaFree(originalImageDevice));
		(cudaFree(intermediateImageDevice));
	}
}


//function to use the CUDA-image filter to apply a guassian filter to the two images
//input images have each pixel stored as an unsigned int in the device (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothImagesAllDataInDevice(unsigned int*& image1InDevice, unsigned int*& image2InDevice, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice, float*& image2SmoothedDevice)
{
	// setup execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_FILTER_IMAGES, BLOCK_SIZE_HEIGHT_FILTER_IMAGES);
	dim3 grid((unsigned int)(ceil((float)widthImages / (float)threads.x)), (unsigned int)(ceil((float)heightImages / (float)threads.y)));

	//copy the width and height of the images to the constant memory of the device
	cudaMemcpyToSymbol(widthImageConstFilt, &widthImages, sizeof(int));
	cudaMemcpyToSymbol(heightImageConstFilt, &heightImages, sizeof(int));

	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH)
	{
		//bind the first input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, image1InDevice, widthImages*heightImages*sizeof(unsigned int));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(image1SmoothedDevice);

		( cudaThreadSynchronize() );

		//now do the same thing with image 2: load to device and convert the pixel values to floats stored in image2SmoothedDevice

		//bind the second input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, image2InDevice, widthImages*heightImages*sizeof(unsigned int));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(image2SmoothedDevice);

		( cudaThreadSynchronize() );


		//unbind the texture the unsigned int textured that the input images were bound to
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
	}

	//otherwise apply a Guassian filter to the images
	else
	{

		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter and the size of the filter to constant memory on the GPU
		cudaMemcpyToSymbol(imageFilterConst, filter, sizeFilter*sizeof(float));
		cudaMemcpyToSymbol(sizeFilterConst, &sizeFilter, sizeof(int));

		//allocate the GPU global memory for the intermediate image (when the image has been filtered horizontally but not vertically)
		float* intermediateImageDevice; 
		(cudaMalloc((void**) &intermediateImageDevice, (widthImages*heightImages*sizeof(float))));

		//first smooth the image 1, so copy image 1 to GPU memory

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, image1InDevice, widthImages*heightImages*sizeof(unsigned int));

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		( cudaThreadSynchronize() );

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (image1SmoothedDevice);

		( cudaThreadSynchronize() );

	

		//then smooth image 2 in the same manner

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, image2InDevice, widthImages*heightImages*sizeof(unsigned int));

		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		( cudaThreadSynchronize() );

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (image2SmoothedDevice);

		( cudaThreadSynchronize() );



		//unbind the texture the unsigned int and float textures used for the smoothing
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
		cudaUnbindTexture( imagePixelsFloatToFilterTexture);

		//free the device memory used to store the intermediate images
		(cudaFree(intermediateImageDevice));
	}
}


//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothSingleImage(unsigned int*& image1InHost, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice)
{
	// setup execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_FILTER_IMAGES, BLOCK_SIZE_HEIGHT_FILTER_IMAGES);
	dim3 grid((unsigned int)(ceil((float)widthImages / (float)threads.x)), (unsigned int)(ceil((float)heightImages / (float)threads.y)));

	//copy the width and height of the images to the constant memory of the device
	cudaMemcpyToSymbol(widthImageConstFilt, &widthImages, sizeof(int));
	cudaMemcpyToSymbol(heightImageConstFilt, &heightImages, sizeof(int));

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

		//bind the first input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, originalImageDevice, widthImages*heightImages*sizeof(unsigned int));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(image1SmoothedDevice);

		( cudaThreadSynchronize() );



		//unbind the texture the unsigned int textured that the input images were bound to
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
	

		//free the device memory used to store original images
		(cudaFree(originalImageDevice));
	}

	//otherwise apply a Guassian filter to the images
	else
	{

		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter and the size of the filter to constant memory on the GPU
		cudaMemcpyToSymbol(imageFilterConst, filter, sizeFilter*sizeof(float));
		cudaMemcpyToSymbol(sizeFilterConst, &sizeFilter, sizeof(int));

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

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, originalImageDevice, widthImages*heightImages*sizeof(unsigned int));

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		( cudaThreadSynchronize() );

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (image1SmoothedDevice);

		( cudaThreadSynchronize() );

		//unbind the texture the unsigned int and float textures used for the smoothing
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
		cudaUnbindTexture( imagePixelsFloatToFilterTexture);

		//free the device memory used to store the images
		(cudaFree(originalImageDevice));
		(cudaFree(intermediateImageDevice));
	}
}


//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned int the device (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothSingleImageAllDataInDevice(unsigned int*& image1InDevice, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice)
{
	// setup execution parameters
	dim3 threads(BLOCK_SIZE_WIDTH_FILTER_IMAGES, BLOCK_SIZE_HEIGHT_FILTER_IMAGES);
	dim3 grid((unsigned int)(ceil((float)widthImages / (float)threads.x)), (unsigned int)(ceil((float)heightImages / (float)threads.y)));

	//copy the width and height of the images to the constant memory of the device
	cudaMemcpyToSymbol(widthImageConstFilt, &widthImages, sizeof(int));
	cudaMemcpyToSymbol(heightImageConstFilt, &heightImages, sizeof(int));

	//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
	//of unsigned ints to an output image of float values
	if (sigmaVal < MIN_SIGMA_VAL_SMOOTH)
	{
		//bind the first input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, image1InDevice, widthImages*heightImages*sizeof(unsigned int));

		//call kernal to convert input unsigned int pixels to output float pixels on the device
		convertUnsignedIntImageToFloat <<< grid, threads >>>(image1SmoothedDevice);

		( cudaThreadSynchronize() );

		//unbind the texture the unsigned int textured that the input images were bound to
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
	
	}

	//otherwise apply a Guassian filter to the images
	else
	{

		//sizeFilter set in makeFilter based on sigmaVal
		int sizeFilter;
		float* filter = makeFilter(sigmaVal, sizeFilter);

		//copy the image filter and the size of the filter to constant memory on the GPU
		cudaMemcpyToSymbol(imageFilterConst, filter, sizeFilter*sizeof(float));
		cudaMemcpyToSymbol(sizeFilterConst, &sizeFilter, sizeof(int));

		float* intermediateImageDevice; 

		//allocate store for intermediate image where images is smoothed in one direction
		(cudaMalloc((void**) &intermediateImageDevice, (widthImages*heightImages*sizeof(float))));

		//first smooth the image 1, so copy image 1 to GPU memory

		//bind the input image on the device to an unsigned int texture
		cudaBindTexture(0, imagePixelsUnsignedIntToFilterTexture, image1InDevice, widthImages*heightImages*sizeof(unsigned int));

		//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
		filterUnsignedIntImageAcross <<< grid, threads >>> (intermediateImageDevice);

		( cudaThreadSynchronize() );

		//bind the "float-valued" intermediate image on the device to a float texture
		cudaBindTexture(0, imagePixelsFloatToFilterTexture, intermediateImageDevice, widthImages*heightImages*sizeof(float));

		//now use the vertical filter to complete the smoothing of image 1 on the device
		filterFloatImageVertical <<< grid, threads >>> (image1SmoothedDevice);

		( cudaThreadSynchronize() );

		//unbind the texture the unsigned int and float textures used for the smoothing
		cudaUnbindTexture( imagePixelsUnsignedIntToFilterTexture);
		cudaUnbindTexture( imagePixelsFloatToFilterTexture);

		//free the device memory used to store the images
		(cudaFree(intermediateImageDevice));
	}
}
