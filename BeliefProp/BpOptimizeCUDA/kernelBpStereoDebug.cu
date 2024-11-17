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

//This file defines CUDA kernel functions for debugging belief propagation processing

template<RunData_t T, unsigned int DISP_VALS>
__global__ void beliefprop::printDataAndMessageValsAtPointKernel(
  unsigned int xVal, unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int widthLevelCheckerboardPart, unsigned int heightLevel)
{
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  }
}


template<RunData_t T, unsigned int DISP_VALS>
__device__ void printDataAndMessageValsAtPointDevice(
  unsigned int xVal, unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int widthLevelCheckerboardPart, unsigned int heightLevel)
{
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  }
}


template<RunData_t T, unsigned int DISP_VALS>
__global__ void beliefprop::printDataAndMessageValsToPointKernel(
  unsigned int xVal, unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int widthLevelCheckerboardPart, unsigned int heightLevel)
{
  const unsigned int checkerboardAdjustment = (((xVal + yVal) % 2) == 0) ? ((yVal)%2) : ((yVal+1)%2);
  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              (xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              (xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  }
}


template<RunData_t T, unsigned int DISP_VALS>
__device__ void printDataAndMessageValsToPointDevice(
  unsigned int xVal, unsigned int yVal,
  T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
  T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
  T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
  T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
  T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
  unsigned int widthLevelCheckerboardPart, unsigned int heightLevel)
{
  const unsigned int checkerboardAdjustment = (((xVal + yVal) % 2) == 0) ? ((yVal)%2) : ((yVal+1)%2);

  if (((xVal + yVal) % 2) == 0) {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              (xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  } else {
    printf("xVal: %d\n", xVal);
    printf("yVal: %d\n", yVal);
    for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
      printf("DISP: %d\n", currentDisparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
          (float) messageUDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
          (float) messageDDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
          (float) messageLDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
          (float) messageRDeviceCurrentCheckerboard0[beliefprop::retrieveIndexInDataAndMessage(
              (xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
          (float) dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
              xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
              currentDisparity, DISP_VALS)]);
    }
  }
}
