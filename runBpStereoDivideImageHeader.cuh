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

//Declares the functions used to run CUDA BP Stereo estimation on an image set where the BP is run on the set in "chunks" rather than all at once (likely
//because there is limited storage on the device)

#ifndef RUN_BP_STEREO_DIVIDE_IMAGE_HEADER_CUH
#define RUN_BP_STEREO_DIVIDE_IMAGE_HEADER_CUH

#include "runBpStereoDivideImageParameters.cuh"

//run the BP Stereo estimation algorithm in "chunks" on a set of images of given width and height stored on the device (already been loaded and "smoothed" if desired)
//assume that the device memory is already allocated for output disparity map in disparityMapImageSetDevice
void runBPStereoEstOnImageSetInChunks(float* image1Device, float* image2Device, unsigned int widthImages, unsigned int heightImages, float*& disparityMapImageSetDevice, BPsettings& runBPAlgSettings, unsigned int widthImageChunk = WIDTH_IMAGE_CHUNK_RUN_STEREO_EST_BP, unsigned int heightImageChunk = HEIGHT_IMAGE_CHUNK_RUN_STEREO_EST_BP, unsigned int imageChunkPaddingX = PADDING_IMAGE_CHUNK_X_RUN_STEREO_EST_BP, unsigned int imageChunkPaddingY = PADDING_IMAGE_CHUNK_Y_RUN_STEREO_EST_BP);

//extract a submatrix widthMatrixRetrieve X heightMatrixRetrieve of size in device memory from a 2d array in the device given the starting point (startXIn2DArray, startYIn2DArray)
//the output outputSubMatrixDevice is a pointer to the desired submatrix in the device 
//the indices and width/height are defined in terms of number of floats "into" the array, not the number of bytes
//it's assumed that the needed device memory for outputSubMatrixDevice has already been allocated
__host__ void extractSubMatrixInDeviceFrom2dArrayInDevice(float*& outputFloatSubMatrixDevice, float*& twoDFloatArrayInDevice, unsigned int startXIn2DArray, unsigned int startYIn2DArray, unsigned int widthOutputMatrix, unsigned int heightOutputMatrix, unsigned int width2DArrayDataFrom, unsigned int height2DArrayDataFrom);

//write a submatrix widthInputMatrixWrite X heightInputMatrixWrite of size in device memory to a 2d array in the device given the starting point (startXIn2DArrayWrite, startYIn2DArrayWrite)
//of the array to write to and the starting point (startXInSubMatrixRead, startYInSubMatrixRead) of the array to read from
//the input inputSubMatrixDevice is a pointer to the submatrix in the device to write to the device memory 2dArrayInDevice at the desired location
//the indices and width/height are defined in terms of number of floats "into" the array, not the number of bytes
__host__ void writeSubMatrixInDeviceTo2dArrayInDevice(float*& inputFloatSubMatrixDevice, float*& twoDFloatArrayInDevice, unsigned int startXInSubMatrixRead, unsigned int startYInSubMatrixRead, unsigned int startXIn2DArrayWrite, unsigned int startYIn2DArrayWrite, unsigned int widthInputMatrixWrite, unsigned int heightInputMatrixWrite, unsigned int totalWidthInputMatrix, unsigned int totalHeightInputMatrix, unsigned int width2DArrayDataTo, unsigned int height2DArrayDataTo);

#endif //RUN_BP_STEREO_DIVIDE_IMAGE_HEADER_CUH

