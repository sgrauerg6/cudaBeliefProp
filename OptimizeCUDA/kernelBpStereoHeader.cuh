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

//This header declares the kernel functions and constant/texture storage to run belief propagation on CUDA

#ifndef KERNEL_BP_H
#define KERNEL_BP_H

#include "ParameterFiles/bpStereoCudaParameters.h"
#include <cuda_fp16.h>

#define PROCESSING_ON_GPU
#include "../SharedFuncts/SharedUtilFuncts.h"
#undef PROCESSING_ON_GPU

//checks if the current point is within the image bounds
__device__ bool withinImageBounds(const unsigned int xVal, const unsigned int yVal, const unsigned int width, const unsigned int height);

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
__device__ int retrieveIndexInDataAndMessage(const unsigned int xVal, const unsigned int yVal, const unsigned int width, const unsigned int height, const unsigned int currentDisparity, const unsigned int totalNumDispVals, const unsigned int offsetData = 0u);

template<BpKernelData_t T>
__device__ T getZeroVal();

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<BpKernelData_t T, unsigned int DISP_VALS>
__device__ void dtStereo(T f[NUM_POSSIBLE_DISPARITY_VALUES]);

// compute current message
template<BpKernelData_t T, unsigned int DISP_VALS>
__device__ void msgStereo(T messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], T messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
  T messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], T dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
  T dst[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp);

//kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void runBPIterationUsingCheckerboardUpdates(const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals);

template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void runBPIterationUsingCheckerboardUpdates(
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals,
  T* dstProcessing);

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void copyPrevLevelToNextLevelBPCheckerboardStereo(const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties currentLevelProperties, const beliefprop::levelProperties nextLevelProperties,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void initializeBottomLevelDataStereo(beliefprop::levelProperties currentLevelProperties,
  float* image1PixelsDevice, float* image2PixelsDevice,
  T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
  const float lambda_bp, const float data_k_bp);

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void initializeCurrentLevelDataStereo(const beliefprop::Checkerboard_Parts checkerboardPart,
  beliefprop::levelProperties currentLevelProperties, beliefprop::levelProperties prevLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1, T* dataCostDeviceToWriteTo,
  const int offsetNum);

//initialize the message values at each pixel of the current level to the default value
template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void initializeMessageValsToDefaultKernel(
  const beliefprop::levelProperties currentLevelProperties,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void retrieveOutputDisparityCheckerboardStereoOptimized(const beliefprop::levelProperties currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice);

template<BpKernelData_t T, unsigned int DISP_VALS>
__device__ void printDataAndMessageValsToPointDevice(const unsigned int xVal, const unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int widthLevelCheckerboardPart, const unsigned int heightLevel);

template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void printDataAndMessageValsToPointKernel(const unsigned int xVal, const unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int widthLevelCheckerboardPart, const unsigned int heightLevel);

template<BpKernelData_t T, unsigned int DISP_VALS>
__device__ void printDataAndMessageValsAtPointDevice(const unsigned int xVal, const unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int widthLevelCheckerboardPart, const unsigned int heightLevel);

template<BpKernelData_t T, unsigned int DISP_VALS>
__global__ void printDataAndMessageValsAtPointKernel(const unsigned int xVal, const unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  const unsigned int widthLevelCheckerboardPart, const unsigned int heightLevel);

#endif //KERNEL_BP_H
