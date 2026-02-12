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
 * @file KernelBpStereoDebug.cu
 * @author Scott Grauer-Gray
 * @brief This file defines CUDA kernel functions for debugging belief propagation processing
 * 
 * @copyright Copyright (c) 2024
 */

template<RunData_t T, unsigned int DISP_VALS>
__global__ void beliefprop::PrintDataAndMessageValsAtPointKernel(
  unsigned int x_val, unsigned int y_val,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int width_level_checkerboard_part, unsigned int heightLevel)
{
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  }
}


template<RunData_t T, unsigned int DISP_VALS>
__device__ void printDataAndMessageValsAtPointDevice(
  unsigned int x_val, unsigned int y_val,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int width_level_checkerboard_part, unsigned int heightLevel)
{
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  }
}


template<RunData_t T, unsigned int DISP_VALS>
__global__ void beliefprop::PrintDataAndMessageValsToPointKernel(
  unsigned int x_val, unsigned int y_val,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int width_level_checkerboard_part, unsigned int heightLevel)
{
  const unsigned int checkerboard_adjustment = (((x_val + y_val) % 2) == 0) ? ((y_val)%2) : ((y_val+1)%2);
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val + 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val - 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2 + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              (x_val / 2 - 1) + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val + 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val - 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2 + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              (x_val / 2 - 1) + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  }
}


template<RunData_t T, unsigned int DISP_VALS>
__device__ void printDataAndMessageValsToPointDevice(
  unsigned int x_val, unsigned int y_val,
  T* data_cost_checkerboard_0, T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int width_level_checkerboard_part, unsigned int heightLevel)
{
  const unsigned int checkerboard_adjustment = (((x_val + y_val) % 2) == 0) ? ((y_val)%2) : ((y_val+1)%2);

  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val + 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val - 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2 + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              (x_val / 2 - 1) + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %d\n", x_val);
    printf("y_val: %d\n", y_val);
    for (size_t current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %d\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) message_u_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val + 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) message_d_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val - 1, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) message_l_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2 + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) message_r_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
              (x_val / 2 - 1) + checkerboard_adjustment, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) data_cost_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
              x_val / 2, y_val, width_level_checkerboard_part, heightLevel,
              current_disparity, DISP_VALS)]);
    }
  }
}
