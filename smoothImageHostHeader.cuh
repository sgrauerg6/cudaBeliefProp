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

//Declares the functions used to smooth the input images with a gaussian filter of SIGMA_BP (see bpCudaParameters.cuh) implemented in CUDA

#ifndef SMOOTH_IMAGE_HOST_HEADER_CUH
#define SMOOTH_IMAGE_HOST_HEADER_CUH

#include "bpStereoCudaParameters.cuh"

//include for the kernal functions to be run on the GPU
#include "kernalFilter.cu"

//functions relating to smoothing the images before running BP

//normalize mask so it integrates to one
__host__ void normalizeFilter(float*& filter, int sizeFilter);

//this function creates a Gaussian filter given a sigma value
__host__ float* makeFilter(float sigma, int& sizeFilter);

//function to use the CUDA-image filter to apply a guassian filter to the two images
__host__ void smoothImages(float*& image1InHost, float*& image2InHost, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice, float*& image2SmoothedDevice);

//function to use the CUDA-image filter to apply a guassian filter to the two images
//input images have each pixel stored as an unsigned int in the device (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothImagesAllDataInDevice(unsigned int*& image1InDevice, unsigned int*& image2InDevice, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice, float*& image2SmoothedDevice);

//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothSingleImage(unsigned int*& image1InHost, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice);

//function to use the CUDA-image filter to apply a guassian filter to the a single images
//input images have each pixel stored as an unsigned int the device (value between 0 and 255 assuming 8-bit grayscale image used)
//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
__host__ void smoothSingleImageAllDataInDevice(unsigned int*& image1InDevice, int widthImages, int heightImages, float sigmaVal, float*& image1SmoothedDevice);

#endif //SMOOTH_IMAGE_HOST_HEADER_CUH
