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

//This header declares the kernal functions and constant/texture storage to run belief propagation on CUDA

#ifndef KERNAL_BP_H
#define KERNAL_BP_H

#include "bpStereoCudaParameters.cuh"


//indexing is performed in such a way so that the memory accesses as coalesced as much as possible
#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION (yVal*width*totalNumDispVals + width*currentDisparity + xVal)
//#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION (height*width*currentDisparity + width*yVal + xVal)
//#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION (width*yVal*totalNumDispVals + xVal*totalNumDispVals + currentDisparity)

//constant memory space to store belief propagation settings
__device__ __constant__ BPsettings BPSettingsConstMemStereo;

// declare texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> image1PixelsTextureBPStereo;
texture<float, 2, cudaReadModeElementType> image2PixelsTextureBPStereo;

#ifdef USE_TEXTURES

texture<float, 1, cudaReadModeElementType> dataCostTexStereoCheckerboard1;
texture<float, 1, cudaReadModeElementType> dataCostTexStereoCheckerboard2;

//when only one set of textures needed, this set is used regardless of actual checkerboard
texture<float, 1, cudaReadModeElementType> messageUPrevTexStereoCheckerboard1;
texture<float, 1, cudaReadModeElementType> messageDPrevTexStereoCheckerboard1;
texture<float, 1, cudaReadModeElementType> messageLPrevTexStereoCheckerboard1;
texture<float, 1, cudaReadModeElementType> messageRPrevTexStereoCheckerboard1;

texture<float, 1, cudaReadModeElementType> messageUPrevTexStereoCheckerboard2;
texture<float, 1, cudaReadModeElementType> messageDPrevTexStereoCheckerboard2;
texture<float, 1, cudaReadModeElementType> messageLPrevTexStereoCheckerboard2;
texture<float, 1, cudaReadModeElementType> messageRPrevTexStereoCheckerboard2;

#endif

//checks if the current point is within the image bounds
__device__ bool withinImageBounds(int xVal, int yVal, int width, int height);


//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
__device__ int retrieveIndexInDataAndMessage(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData = 0);


//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
__device__ void dtStereo(float f[NUM_POSSIBLE_DISPARITY_VALUES]);


// compute current message
__device__ void msgStereo(float messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], float messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES], 
	float messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], float dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
	float dst[NUM_POSSIBLE_DISPARITY_VALUES]);


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
__global__ void initializeBottomLevelDataStereo(float* dataCostDeviceStereoCheckerboard1, float* dataCostDeviceStereoCheckerboard2);


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
__global__ void initializeCurrentLevelDataStereo(float* dataCostDeviceToWriteTo, int widthLevel, int heightLevel, int checkerboardPart, int offsetNum);


//initialize the message values at each pixel of the current level to the default value
__global__ void initializeMessageValsToDefault(float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1, float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, float* messageLDeviceCurrentCheckerboard2, float* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel);


//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
__device__ void runBPIterationInOutDataInLocalMem(float prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES], float prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES], float prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES], float prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES], float dataMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES], float currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES]);


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the 
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
__device__ void runBPIterationUsingCheckerboardUpdatesDeviceUseTexBoundAndLocalMem(float* messageUDeviceCurrentCheckerboardOut, float* messageDDeviceCurrentCheckerboardOut, float* messageLDeviceCurrentCheckerboardOut, float* messageRDeviceCurrentCheckerboardOut, int widthLevelCheckerboardPart, int heightLevel, int iterationNum, int xVal, int yVal, int offsetData);


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard" 
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
__global__ void runBPIterationUsingCheckerboardUpdates(float* messageUDeviceCurrentCheckerboardOut, float* messageDDeviceCurrentCheckerboardOut, float* messageLDeviceCurrentCheckerboardOut, float* messageRDeviceCurrentCheckerboardOut, int widthLevel, int heightLevel, int iterationNum, int offsetData);


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
__global__ void copyPrevLevelToNextLevelBPCheckerboardStereo(float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1, float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, float* messageLDeviceCurrentCheckerboard2, float* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPart);


//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
__global__ void retrieveOutputDisparityCheckerboardStereo(float* disparityBetweenImagesDevice, int widthLevel, int heightLevel);


#endif //KERNAL_BP_H
