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

//This file defines the methods to perform belief propagation for disparity map estimation from stereo images on CUDA

#include "BpConstsAndParams/BpStereoCudaParameters.h"
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "kernelBpStereoHalf.cu"

//uncomment for CUDA kernel debug functions for belief propagation processing
//#include "kernelBpStereoDebug.h"

namespace beliefpropCUDA {

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<RunData_t T, unsigned int DISP_VALS>
__global__ void initializeBottomLevelData(
  beliefprop::LevelProperties currentLevelProperties,
  float* image1PixelsDevice, float* image2PixelsDevice,
  T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
  float lambda_bp, float data_k_bp, unsigned int bpSettingsDispVals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  //get the x value within the current "checkerboard"
  const unsigned int xInCheckerboard = xVal / 2;

  if (run_imp_util::withinImageBounds(xInCheckerboard, yVal, currentLevelProperties.width_level_, currentLevelProperties.height_level_))
  {
    beliefprop::initializeBottomLevelDataPixel<T, DISP_VALS>(xVal, yVal,
      currentLevelProperties, image1PixelsDevice,
      image2PixelsDevice, dataCostDeviceStereoCheckerboard0,
      dataCostDeviceStereoCheckerboard1, lambda_bp,
      data_k_bp, bpSettingsDispVals);
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, unsigned int DISP_VALS>
__global__ void initializeCurrentLevelData(
  beliefprop::Checkerboard_Part checkerboardPart,
  beliefprop::LevelProperties currentLevelProperties,
  beliefprop::LevelProperties prevLevelProperties, T* dataCostStereoCheckerboard0,
  T* dataCostStereoCheckerboard1, T* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bpSettingsDispVals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::withinImageBounds(xVal, yVal, currentLevelProperties.width_checkerboard_level_, currentLevelProperties.height_level_))
  {
    beliefprop::initializeCurrentLevelDataPixel<T, T, DISP_VALS>(
      xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
      offsetNum, bpSettingsDispVals);
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
__global__ void initializeMessageValsToDefaultKernel(
  beliefprop::LevelProperties currentLevelProperties,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bpSettingsDispVals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int xValInCheckerboard = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::withinImageBounds(xValInCheckerboard, yVal, currentLevelProperties.width_checkerboard_level_, currentLevelProperties.height_level_))
  {
    //initialize message values in both checkerboards
    beliefprop::initializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
      xValInCheckerboard,  yVal, currentLevelProperties,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      bpSettingsDispVals);
  }
}

//kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<RunData_t T, unsigned int DISP_VALS>
__global__ void runBPIterationUsingCheckerboardUpdates(
  beliefprop::Checkerboard_Part checkerboardToUpdate, beliefprop::LevelProperties currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::withinImageBounds(xVal, yVal, currentLevelProperties.width_level_/2, currentLevelProperties.height_level_))
  {
    beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      xVal, yVal, checkerboardToUpdate, currentLevelProperties,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      disc_k_bp, 0, dataAligned, bpSettingsDispVals);
  }
}

template<RunData_t T, unsigned int DISP_VALS>
__global__ void runBPIterationUsingCheckerboardUpdates(
  beliefprop::Checkerboard_Part checkerboardToUpdate, beliefprop::LevelProperties currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, bool dataAligned, unsigned int bpSettingsDispVals,
  void* dstProcessing)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::withinImageBounds(xVal, yVal, currentLevelProperties.width_level_/2, currentLevelProperties.height_level_))
  {
    beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      xVal, yVal, checkerboardToUpdate, currentLevelProperties,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      disc_k_bp, 0, dataAligned, bpSettingsDispVals, dstProcessing);
  }
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
__global__ void copyMsgDataToNextLevel(
  beliefprop::Checkerboard_Part checkerboardPart,
  beliefprop::LevelProperties currentLevelProperties,
  beliefprop::LevelProperties nextLevelProperties,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bpSettingsDispVals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::withinImageBounds(xVal, yVal, currentLevelProperties.width_checkerboard_level_, currentLevelProperties.height_level_))
  {
    beliefprop::copyMsgDataToNextLevelPixel<T, DISP_VALS>(xVal, yVal,
      checkerboardPart, currentLevelProperties, nextLevelProperties,
      messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
      messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
      messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
      messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      bpSettingsDispVals);
  }
}

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<RunData_t T, unsigned int DISP_VALS>
__global__ void retrieveOutputDisparity(
  beliefprop::LevelProperties currentLevelProperties,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bpSettingsDispVals)
{
  //get x and y indices for the current CUDA thread
  const unsigned int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::withinImageBounds(xVal, yVal, currentLevelProperties.width_checkerboard_level_, currentLevelProperties.height_level_))
  {
    beliefprop::retrieveOutputDisparityPixel<T, T, DISP_VALS>(
      xVal, yVal, currentLevelProperties,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
      messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
      messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
      messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
      messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
      disparityBetweenImagesDevice, bpSettingsDispVals);
  }
}

};
