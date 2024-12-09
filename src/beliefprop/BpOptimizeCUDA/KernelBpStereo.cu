/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file KernelBpStereo.cu
 * @author Scott Grauer-Gray
 * @brief This file defines the methods to perform belief propagation
 * for disparity map estimation from stereo images on CUDA
 * 
 * @copyright Copyright (c) 2024
 */

#include "BpSharedFuncts/SharedBpProcessingFuncts.h"

//uncomment to set CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF (disabled by default) since that could
//get overflow in message values during processing in some bp settings (only happened on largest stereo set in testing)
//shouldn't be needed if using bfloat since that has a higher exponent and not likely to overflow
//recommend using bfloat rather than enabling this setting if target GPU supports bflow
//#define CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

#ifdef CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF
#include "KernelBpStereoHalf.cu"
#endif //CHECK_VAL_TO_NORMALIZE_VALID_CUDA_HALF

//uncomment for CUDA kernel debug functions for belief propagation processing
//#include "KernelBpStereoDebug.h"

/**
 * @brief Namespace to define global kernel functions for parallel belief propagation
 * processing using CUDA.
 * 
 */
namespace beliefprop_cuda {

/**
 * @brief Initialize the "data cost" for each possible disparity between the
 * two full-sized input images ("bottom" of the image pyramid).
 * The image data is stored in the image_1_pixels_device and
 * image_2_pixels_device arrays.
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param current_bp_level 
 * @param image_1_pixels_device 
 * @param image_2_pixels_device 
 * @param data_cost_stereo_checkerboard_0 
 * @param data_cost_stereo_checkerboard_1 
 * @param lambda_bp 
 * @param data_k_bp 
 * @param bp_settings_disp_vals 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void InitializeBottomLevelData(
  beliefprop::BpLevelProperties current_bp_level,
  float* image_1_pixels_device, float* image_2_pixels_device,
  T* data_cost_stereo_checkerboard_0, T* data_cost_stereo_checkerboard_1,
  float lambda_bp, float data_k_bp, unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  //get the x value within the current "checkerboard"
  const unsigned int x_checkerboard = x_val / 2;

  if (beliefprop::WithinImageBounds(
    x_checkerboard, y_val, current_bp_level.width_level_, current_bp_level.height_level_))
  {
    beliefprop::InitializeBottomLevelDataPixel<T, DISP_VALS>(x_val, y_val,
      current_bp_level, image_1_pixels_device,
      image_2_pixels_device, data_cost_stereo_checkerboard_0,
      data_cost_stereo_checkerboard_1, lambda_bp,
      data_k_bp, bp_settings_disp_vals);
  }
}

/**
 * @brief Initialize the data costs at the "next" level up in the pyramid given
 * that the data at the lower has been set
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param checkerboard_part 
 * @param current_bp_level 
 * @param prev_bp_level 
 * @param data_cost_checkerboard_0 
 * @param data_cost_checkerboard_1 
 * @param data_cost_current_level 
 * @param offset_num 
 * @param bp_settings_disp_vals 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void InitializeCurrentLevelData(
  beliefprop::CheckerboardPart checkerboard_part,
  beliefprop::BpLevelProperties current_bp_level,
  beliefprop::BpLevelProperties prev_bp_level, T* data_cost_checkerboard_0,
  T* data_cost_checkerboard_1, T* data_cost_current_level,
  unsigned int offset_num, unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (beliefprop::WithinImageBounds(
    x_val, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    beliefprop::InitializeCurrentLevelDataPixel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_part, current_bp_level, prev_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1, data_cost_current_level,
      offset_num, bp_settings_disp_vals);
  }
}

/**
 * @brief Initialize the message values at each pixel of the current level to the default value
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param current_bp_level 
 * @param message_u_checkerboard_0 
 * @param message_d_checkerboard_0 
 * @param message_l_checkerboard_0 
 * @param message_r_checkerboard_0 
 * @param message_u_checkerboard_1 
 * @param message_d_checkerboard_1 
 * @param message_l_checkerboard_1 
 * @param message_r_checkerboard_1 
 * @param bp_settings_disp_vals 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void InitializeMessageValsToDefaultKernel(
  beliefprop::BpLevelProperties current_bp_level,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val_in_checkerboard = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (beliefprop::WithinImageBounds(
    x_val_in_checkerboard, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    //initialize message values in both checkerboards
    beliefprop::InitializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
      x_val_in_checkerboard,  y_val, current_bp_level,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      bp_settings_disp_vals);
  }
}

/**
 * @brief Kernel function to run the current iteration of belief propagation
 * in parallel using the checkerboard update method where half the pixels in
 * the "checkerboard" scheme retrieve messages from each 4-connected neighbor
 * and then update their message based on the retrieved messages and the data cost
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param checkerboard_to_update 
 * @param current_bp_level 
 * @param data_cost_checkerboard_0 
 * @param data_cost_checkerboard_1 
 * @param message_u_checkerboard_0 
 * @param message_d_checkerboard_0 
 * @param message_l_checkerboard_0 
 * @param message_r_checkerboard_0 
 * @param message_u_checkerboard_1 
 * @param message_d_checkerboard_1 
 * @param message_l_checkerboard_1 
 * @param message_r_checkerboard_1 
 * @param disc_k_bp 
 * @param data_aligned 
 * @param bp_settings_disp_vals 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void RunBPIterationUsingCheckerboardUpdates(
  beliefprop::CheckerboardPart checkerboard_to_update, beliefprop::BpLevelProperties current_bp_level,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (beliefprop::WithinImageBounds(
    x_val, y_val, current_bp_level.width_level_/2, current_bp_level.height_level_))
  {
    beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_to_update, current_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disc_k_bp, 0, data_aligned, bp_settings_disp_vals);
  }
}

/**
 * @brief Kernel function to run the current iteration of belief propagation
 * in parallel using the checkerboard update method where half the pixels in
 * the "checkerboard" scheme retrieve messages from each 4-connected neighbor
 * and then update their message based on the retrieved messages and the data cost.
 * Function differs from counterpart overloaded function with same name in that
 * it takes in allocated memory to use in processing.
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param checkerboard_to_update 
 * @param current_bp_level 
 * @param data_cost_checkerboard_0 
 * @param data_cost_checkerboard_1 
 * @param message_u_checkerboard_0 
 * @param message_d_checkerboard_0 
 * @param message_l_checkerboard_0 
 * @param message_r_checkerboard_0 
 * @param message_u_checkerboard_1 
 * @param message_d_checkerboard_1 
 * @param message_l_checkerboard_1 
 * @param message_r_checkerboard_1 
 * @param disc_k_bp 
 * @param data_aligned 
 * @param bp_settings_disp_vals 
 * @param dst_processing 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void RunBPIterationUsingCheckerboardUpdates(
  beliefprop::CheckerboardPart checkerboard_to_update, beliefprop::BpLevelProperties current_bp_level,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals,
  void* dst_processing)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (beliefprop::WithinImageBounds(
    x_val, y_val, current_bp_level.width_level_/2, current_bp_level.height_level_))
  {
    beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_to_update, current_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disc_k_bp, 0, data_aligned, bp_settings_disp_vals, dst_processing);
  }
}

/**
 * @brief Kernel to copy the computed BP message values at the current level to the
 * corresponding locations at the "next" level down. The kernel works from the point
 * of view of the pixel at the current level that is being copied to four corresponding
 * pixels in the next level.
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param checkerboard_part 
 * @param current_bp_level 
 * @param next_bp_level 
 * @param message_u_prev_checkerboard_0 
 * @param message_d_prev_checkerboard_0 
 * @param message_l_prev_checkerboard_0 
 * @param message_r_prev_checkerboard_0 
 * @param message_u_prev_checkerboard_1 
 * @param message_d_prev_checkerboard_1 
 * @param message_l_prev_checkerboard_1 
 * @param message_r_prev_checkerboard_1 
 * @param message_u_checkerboard_0 
 * @param message_d_checkerboard_0 
 * @param message_l_checkerboard_0 
 * @param message_r_checkerboard_0 
 * @param message_u_checkerboard_1 
 * @param message_d_checkerboard_1 
 * @param message_l_checkerboard_1 
 * @param message_r_checkerboard_1 
 * @param bp_settings_disp_vals 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void CopyMsgDataToNextLevel(
  beliefprop::CheckerboardPart checkerboard_part,
  beliefprop::BpLevelProperties current_bp_level,
  beliefprop::BpLevelProperties next_bp_level,
  T* message_u_prev_checkerboard_0, T* message_d_prev_checkerboard_0,
  T* message_l_prev_checkerboard_0, T* message_r_prev_checkerboard_0,
  T* message_u_prev_checkerboard_1, T* message_d_prev_checkerboard_1,
  T* message_l_prev_checkerboard_1, T* message_r_prev_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals)
{
  //get the x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (beliefprop::WithinImageBounds(
    x_val, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    beliefprop::CopyMsgDataToNextLevelPixel<T, DISP_VALS>(x_val, y_val,
      checkerboard_part, current_bp_level, next_bp_level,
      message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
      message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
      message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
      message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      bp_settings_disp_vals);
  }
}

/**
 * @brief Retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @param current_bp_level 
 * @param data_cost_checkerboard_0 
 * @param data_cost_checkerboard_1 
 * @param message_u_checkerboard_0 
 * @param message_d_checkerboard_0 
 * @param message_l_checkerboard_0 
 * @param message_r_checkerboard_0 
 * @param message_u_checkerboard_1 
 * @param message_d_checkerboard_1 
 * @param message_l_checkerboard_1 
 * @param message_r_checkerboard_1 
 * @param disparity_between_images_device 
 * @param bp_settings_disp_vals 
 * @return __global__ 
 */
template<RunData_t T, unsigned int DISP_VALS>
__global__ void RetrieveOutputDisparity(
  beliefprop::BpLevelProperties current_bp_level,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals)
{
  //get x and y indices for the current CUDA thread
  const unsigned int x_val = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y_val = blockIdx.y * blockDim.y + threadIdx.y;

  if (beliefprop::WithinImageBounds(
    x_val, y_val, current_bp_level.width_checkerboard_level_, current_bp_level.height_level_))
  {
    beliefprop::RetrieveOutputDisparityPixel<T, T, DISP_VALS>(
      x_val, y_val, current_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disparity_between_images_device, bp_settings_disp_vals);
  }
}

};
