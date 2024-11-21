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

#include "BpSharedFuncts/SharedBPProcessingFuncts.h"

//uncomment to set CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF (disabled by default) since that could
//get overflow in message values during processing in some bp settings (only happened on largest stereo set in testing)
//shouldn't be needed if using bfloat since that has a higher exponent and not likely to overflow
//recommend using bfloat rather than enabling this setting if target GPU supports bflow
//#define CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

#ifdef CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF
#include "kernelBpStereoHalf.cu"
#endif //CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

//uncomment for CUDA kernel debug functions for belief propagation processing
//#include "kernelBpStereoDebug.h"

namespace beliefpropCUDA {

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<RunData_t T, unsigned int DISP_VALS>
__global__ void InitializeBottomLevelData(
  beliefprop::BpLevelProperties current_bp_level,
  float* image_1_pixels_device, float* image_2_pixels_device,
  T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
  float lambda_bp, float data_k_bp, unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  //get the x value within the current "checkerboard"
  const unsigned int xInCheckerboard = x_val / 2;

  if (run_imp_util::WithinImageBounds(
    xInCheckerboard, y_val, current_bp_level.width_level_, current_bp_level.height_level_))
  {
    beliefprop::InitializeBottomLevelDataPixel<T, DISP_VALS>(x_val, y_val,
      current_bp_level, image_1_pixels_device,
      image_2_pixels_device, dataCostDeviceStereoCheckerboard0,
      dataCostDeviceStereoCheckerboard1, lambda_bp,
      data_k_bp, bp_settings_disp_vals);
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, unsigned int DISP_VALS>
__global__ void InitializeCurrentLevelData(
  beliefprop::CheckerboardPart checkerboard_part,
  beliefprop::BpLevelProperties current_bp_level,
  beliefprop::BpLevelProperties prev_bp_level, T* dataCostStereoCheckerboard0,
  T* dataCostStereoCheckerboard1, T* dataCostDeviceToWriteTo,
  unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::WithinImageBounds(
    x_val, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    beliefprop::InitializeCurrentLevelDataPixel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
      offset_num, bp_settings_disp_vals);
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
__global__ void InitializeMessageValsToDefaultKernel(
  beliefprop::BpLevelProperties current_bp_level,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val_in_checkerboard = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::WithinImageBounds(
    x_val_in_checkerboard, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    //initialize message values in both checkerboards
    beliefprop::InitializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
      x_val_in_checkerboard,  y_val, current_bp_level,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      bp_settings_disp_vals);
  }
}

//kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<RunData_t T, unsigned int DISP_VALS>
__global__ void RunBPIterationUsingCheckerboardUpdates(
  beliefprop::CheckerboardPart checkerboard_to_update, beliefprop::BpLevelProperties current_bp_level,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::WithinImageBounds(
    x_val, y_val, current_bp_level.width_level_/2, current_bp_level.height_level_))
  {
    beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_to_update, current_bp_level,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      disc_k_bp, 0, data_aligned, bp_settings_disp_vals);
  }
}

template<RunData_t T, unsigned int DISP_VALS>
__global__ void RunBPIterationUsingCheckerboardUpdates(
  beliefprop::CheckerboardPart checkerboard_to_update, beliefprop::BpLevelProperties current_bp_level,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  void* dstProcessing)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::WithinImageBounds(
    x_val, y_val, current_bp_level.width_level_/2, current_bp_level.height_level_))
  {
    beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_to_update, current_bp_level,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      disc_k_bp, 0, data_aligned, bp_settings_disp_vals, dstProcessing);
  }
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
__global__ void CopyMsgDataToNextLevel(
  beliefprop::CheckerboardPart checkerboard_part,
  beliefprop::BpLevelProperties current_bp_level,
  beliefprop::BpLevelProperties next_bp_level,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::WithinImageBounds(
    x_val, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    beliefprop::CopyMsgDataToNextLevelPixel<T, DISP_VALS>(x_val, y_val,
      checkerboard_part, current_bp_level, next_bp_level,
      messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
      messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
      messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
      messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
      messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
      messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
      messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
      messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
      bp_settings_disp_vals);
  }
}

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<RunData_t T, unsigned int DISP_VALS>
__global__ void RetrieveOutputDisparity(
  beliefprop::BpLevelProperties current_bp_level,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
  T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
  T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
  T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  //get x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (run_imp_util::WithinImageBounds(
    x_val, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    beliefprop::RetrieveOutputDisparityPixel<T, T, DISP_VALS>(
      x_val, y_val, current_bp_level,
      dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
      messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
      messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
      messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
      messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
      disparity_between_images_device, bp_settings_disp_vals);
  }
}

};
