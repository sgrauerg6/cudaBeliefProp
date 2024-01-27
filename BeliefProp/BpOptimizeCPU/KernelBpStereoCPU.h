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

#ifndef KERNEL_BP_STEREO_CPU_H
#define KERNEL_BP_STEREO_CPU_H

#include <math.h>
#include <omp.h>
#include <algorithm>
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpConstsAndParams/bpStructsAndEnums.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"

class KernelBpStereoCPU
{
public:

  //initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
  //the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
  template<RunData_t T, unsigned int DISP_VALS>
  static void initializeBottomLevelDataStereoCPU(const beliefprop::levelProperties& currentLevelProperties,
    float* image1PixelsDevice, float* image2PixelsDevice,
    T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
    const float lambda_bp, const float data_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<RunData_t T, unsigned int DISP_VALS>
  static void initializeCurrentLevelDataStereoCPU(const beliefprop::Checkerboard_Parts checkerboardPart,
    const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //initialize the message values at each pixel of the current level to the default value
  template<RunData_t T, unsigned int DISP_VALS>
  static void initializeMessageValsToDefaultKernelCPU(const beliefprop::levelProperties& currentLevelProperties,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
  //scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
  template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
  static void runBPIterationUsingCheckerboardUpdatesCPU(const beliefprop::Checkerboard_Parts checkerboardToUpdate,
    const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsNumDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<RunData_t T, unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions(
    const beliefprop::Checkerboard_Parts checkerboardPartUpdate, const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
  //the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
  template<RunData_t T, unsigned int DISP_VALS>
  static void copyPrevLevelToNextLevelBPCheckerboardStereoCPU(const beliefprop::Checkerboard_Parts checkerboardPart,
    const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& nextLevelProperties,
    T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
    T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
    T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
    T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
  template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPU(const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
    T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
    T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
    T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel using SIMD vectors
  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, unsigned int DISP_VALS>
  static void retrieveOutDispOptimizedCPUUseSIMDVectorsProcess(const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
    T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
    T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
    T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const unsigned int numDataInSIMDVector,
    const beliefprop::ParallelParameters& optCPUParams);
  
  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsAVX256(
    const beliefprop::levelProperties& currentLevelProperties,
    float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
    float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
    float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
    float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
    float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsAVX256(
    const beliefprop::levelProperties& currentLevelProperties,
    short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
    short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
    short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
    short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
    short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsAVX256(
    const beliefprop::levelProperties& currentLevelProperties,
    double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
    double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
    double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
    double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
    double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsAVX512(
    const beliefprop::levelProperties& currentLevelProperties,
    float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
    float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
    float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
    float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
    float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsAVX512(
    const beliefprop::levelProperties& currentLevelProperties,
    short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
    short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
    short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0,
    short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
    short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsAVX512(
    const beliefprop::levelProperties& currentLevelProperties,
    double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
    double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
    double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
    double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
    double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsNEON(
    const beliefprop::levelProperties& currentLevelProperties,
    float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
    float* messageUPrevStereoCheckerboard0, float* messageDPrevStereoCheckerboard0,
    float* messageLPrevStereoCheckerboard0, float* messageRPrevStereoCheckerboard0,
    float* messageUPrevStereoCheckerboard1, float* messageDPrevStereoCheckerboard1,
    float* messageLPrevStereoCheckerboard1, float* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsNEON(
    const beliefprop::levelProperties& currentLevelProperties,
    double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
    double* messageUPrevStereoCheckerboard0, double* messageDPrevStereoCheckerboard0,
    double* messageLPrevStereoCheckerboard0, double* messageRPrevStereoCheckerboard0,
    double* messageUPrevStereoCheckerboard1, double* messageDPrevStereoCheckerboard1,
    double* messageLPrevStereoCheckerboard1, double* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

#ifdef COMPILING_FOR_ARM
  template<unsigned int DISP_VALS>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectorsNEON(
    const beliefprop::levelProperties& currentLevelProperties,
    float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
    float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
    float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
    float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
    float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);
#endif //COMPILING_FOR_ARM

  //device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
  //and the output message values are save to output message arrays
  template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
  static void runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(const unsigned int xValStartProcessing,
    const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
    U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
    U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS],
    U dataMessage[DISP_VALS],
    T* currentUMessageArray, T* currentDMessageArray,
    T* currentLMessageArray, T* currentRMessageArray,
    const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing);

  template<RunData_t T, RunDataVect_t U>
  static void runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(
    const unsigned int xValStartProcessing, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U* prevUMessage, U* prevDMessage,
    U* prevLMessage, U* prevRMessage,
    U* dataMessage,
    T* currentUMessageArray, T* currentDMessageArray,
    T* currentLMessageArray, T* currentRMessageArray,
    const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing,
    const unsigned int bpSettingsDispVals);
  
  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsAVX256(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
    float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
    float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
    float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
    float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsAVX256(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
    short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
    short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
    short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
    short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsAVX256(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
    double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
    double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
    double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
    double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);
  
  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsAVX512(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
    float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
    float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
    float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
    float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsAVX512(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
    short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
    short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
    short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
    short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsAVX512(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
    double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
    double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
    double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
    double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsNEON(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
    float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
    float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
    float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
    float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsNEON(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
    double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
    double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
    double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
    double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

#ifdef COMPILING_FOR_ARM
  template<unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsNEON(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
    float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
    float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
    float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
    float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);
#endif //COMPILING_FOR_ARM

  template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
  static void runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess(
    const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int numDataInSIMDVector,
    const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  // compute current message
  template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
  static void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
    U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned);

  // compute current message
  template<RunData_t T, RunDataVect_t U>
  static void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U* messageValsNeighbor1, U* messageValsNeighbor2,
    U* messageValsNeighbor3, U* dataCosts,
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
    const unsigned int bpSettingsDispVals);

  // compute current message
  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W>
  static void msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U* messageValsNeighbor1, U* messageValsNeighbor2,
    U* messageValsNeighbor3, U* dataCosts,
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
    const unsigned int bpSettingsDispVals);

  // compute current message
  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, unsigned int DISP_VALS>
  static void msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
    U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned);

  //function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  template<RunDataProcess_t T, RunDataVectProcess_t U, unsigned int DISP_VALS>
  static void dtStereoSIMD(U f[DISP_VALS]);

  //function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  //TODO: look into defining function in .cpp file so don't need to declare inline
  template<RunDataProcess_t T, RunDataVectProcess_t U>
  static void dtStereoSIMD(U* f, const unsigned int bpSettingsDispVals);

  template<RunDataVectProcess_t T>
  static void updateBestDispBestVals(T& bestDisparities, T& bestVals, const T& currentDisparity, const T& valAtDisp) {
    printf("Data type not supported for updating best disparities and values\n");
  }

  template<RunData_t T, unsigned int DISP_VALS>
  static void printDataAndMessageValsAtPointKernelCPU(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

  template<RunData_t T, unsigned int DISP_VALS>
  static void printDataAndMessageValsToPointKernelCPU(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);
};

//headers to include differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
#include "KernelBpStereoCPU_ARMTemplateSpFuncts.h"

#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
#include "KernelBpStereoCPU_NEON.h"
#endif //CPU_VECTORIZATION_DEFINE == NEON_DEFINE

#else
//needed so that template specializations are used when available
#include "KernelBpStereoCPU_TemplateSpFuncts.h"

#if (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"
#include "KernelBpStereoCPU_AVX512TemplateSpFuncts.h"
#endif

#endif //COMPILING_FOR_ARM

#endif //KERNEL_BP_STEREO_CPU_H
