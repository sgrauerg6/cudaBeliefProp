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

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../ParameterFiles/bpRunSettings.h"
#include <math.h>
#if ((CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS) || (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED))
#include "../ThreadPool/thread_pool.hpp"
#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
#include <omp.h>
#endif //CPU_PARALLELIZATION_METHOD
#include <algorithm>

//headers to include differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h>
#endif //COMPILING_FOR_ARM

//define concepts of allowed data types for belief propagation data storage and processing
#ifdef COMPILING_FOR_ARM
//float16_t is used for half data type in ARM processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, float16_t>;
#else
//short is used for half data type in x86 processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, short>;
#endif //COMPILING_FOR_ARM

template <typename T>
concept BpDataProcess_t = std::is_same_v<T, float> || std::is_same_v<T, double>;

//SIMD types differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
template <typename T>
concept BpDataVect_t = std::is_same_v<T, float64x2_t> || std::is_same_v<T, float32x4_t> || std::is_same_v<T, float16x4_t>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, float64x2_t> || std::is_same_v<T, float32x4_t>;
#else
#if (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
template <typename T>
concept BpDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256>;
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
template <typename T>
concept BpDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512> || std::is_same_v<T, __m256i>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512>;
#endif //CPU_VECTORIZATION_DEFINE
#endif //COMPILING_FOR_ARM

//concepts that allow both single and vectorized types
template <typename T>
concept BpDataSingOrVect_t = BpData_t<T> || BpDataVect_t<T>;

template <typename T>
concept BpDataProcessSingOrVect_t = BpDataProcess_t<T> || BpDataVectProcess_t<T>;

class KernelBpStereoCPU
{
public:
#if ((CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS) || (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED))
  //initialize thread pool with default number of threads
  inline static thread_pool tPool;
#endif //CPU_PARALLELIZATION_METHOD

  //initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
  //the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
  template<BpData_t T, unsigned int DISP_VALS>
  static void initializeBottomLevelDataStereoCPU(const beliefprop::levelProperties& currentLevelProperties,
    float* image1PixelsDevice, float* image2PixelsDevice,
    T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
    const float lambda_bp, const float data_k_bp, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<BpData_t T, unsigned int DISP_VALS>
  static void initializeCurrentLevelDataStereoCPU(const beliefprop::Checkerboard_Parts checkerboardPart,
    const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //initialize the message values at each pixel of the current level to the default value
  template<BpData_t T, unsigned int DISP_VALS>
  static void initializeMessageValsToDefaultKernelCPU(const beliefprop::levelProperties& currentLevelProperties,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
  //scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
  template<BpData_t T, unsigned int DISP_VALS, beliefprop::AccSetting VECTORIZATION>
  static void runBPIterationUsingCheckerboardUpdatesCPU(const beliefprop::Checkerboard_Parts checkerboardToUpdate,
    const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
    const float disc_k_bp, const unsigned int bpSettingsNumDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  template<BpData_t T, unsigned int DISP_VALS>
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
  template<BpData_t T, unsigned int DISP_VALS>
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
  template<BpData_t T, unsigned int DISP_VALS, beliefprop::AccSetting VECTORIZATION>
  static void retrieveOutputDisparityCheckerboardStereoOptimizedCPU(const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
    T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
    T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
    T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
    float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
    const beliefprop::ParallelParameters& optCPUParams);

  //retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel using SIMD vectors
  template<BpData_t T, BpDataVect_t U, BpDataProcess_t V, BpDataVectProcess_t W, unsigned int DISP_VALS>
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
  template<BpData_t T, BpDataVect_t U, unsigned int DISP_VALS>
  static void runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(const unsigned int xValStartProcessing,
    const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
    U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
    U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS],
    U dataMessage[DISP_VALS],
    T* currentUMessageArray, T* currentDMessageArray,
    T* currentLMessageArray, T* currentRMessageArray,
    const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing);

  template<BpData_t T, BpDataVect_t U>
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

  template<BpData_t T, BpDataVect_t U, unsigned int DISP_VALS>
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
  template<BpData_t T, BpDataVect_t U, unsigned int DISP_VALS>
  static void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
    U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned);

  // compute current message
  template<BpData_t T, BpDataVect_t U>
  static void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U* messageValsNeighbor1, U* messageValsNeighbor2,
    U* messageValsNeighbor3, U* dataCosts,
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
    const unsigned int bpSettingsDispVals);

  // compute current message
  template<BpData_t T, BpDataVect_t U, BpDataProcess_t V, BpDataVectProcess_t W>
  static void msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U* messageValsNeighbor1, U* messageValsNeighbor2,
    U* messageValsNeighbor3, U* dataCosts,
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
    const unsigned int bpSettingsDispVals);

  // compute current message
  template<BpData_t T, BpDataVect_t U, BpDataProcess_t V, BpDataVectProcess_t W, unsigned int DISP_VALS>
  static void msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
    U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
    T* dstMessageArray, const U& disc_k_bp, const bool dataAligned);

  //function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  template<BpDataProcess_t T, BpDataVectProcess_t U, unsigned int DISP_VALS>
  static void dtStereoSIMD(U f[DISP_VALS]);

  //function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  //TODO: look into defining function in .cpp file so don't need to declare inline
  template<BpDataProcess_t T, BpDataVectProcess_t U>
  static void dtStereoSIMD(U* f, const unsigned int bpSettingsDispVals);

  template<BpDataVectProcess_t T>
  static void updateBestDispBestVals(T& bestDisparities, T& bestVals, const T& currentDisparity, const T& valAtDisp) {
    printf("Data type not supported for updating best disparities and values\n");
  }

  template<BpData_t T, BpDataVect_t U>
  static U loadPackedDataAligned(const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
    const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, T* inData)
  {
    printf("Data type not supported for loading aligned data\n");
  }

  template<BpData_t T, BpDataVect_t U>
  static U loadPackedDataUnaligned(const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
    const beliefprop::levelProperties& currentLevelProperties, const unsigned int numDispVals, T* inData)
  {
    printf("Data type not supported for loading unaligned data\n");
  }

  template<BpDataVect_t T>
  static T createSIMDVectorSameData(const float data) {
    printf("Data type not supported for creating simd vector\n");
  }

  template<BpDataSingOrVect_t T, BpDataSingOrVect_t U, BpDataSingOrVect_t V>
  static V addVals(const T& val1, const U& val2) { return (val1 + val2); }

  template<BpDataSingOrVect_t T, BpDataSingOrVect_t U, BpDataSingOrVect_t V>
  static V subtractVals(const T& val1, const U& val2) { return (val1 - val2); }

  template<BpDataSingOrVect_t T, BpDataSingOrVect_t U, BpDataSingOrVect_t V>
  static V divideVals(const T& val1, const U& val2) { return (val1 / val2); }

  template<BpDataSingOrVect_t T, BpDataSingOrVect_t V>
  static T convertValToDatatype(const V val) { return (T)val; }

  template<BpDataSingOrVect_t T>
  static T getMinByElement(const T& val1, const T& val2) { return std::min(val1, val2); }

  template<BpData_t T, BpDataVectProcess_t U>
  static void storePackedDataAligned(const unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }

  template<BpData_t T, BpDataVectProcess_t U>
  static void storePackedDataUnaligned(const unsigned int indexDataStore, T* locationDataStore, const U& dataToStore) {
    locationDataStore[indexDataStore] = dataToStore;
  }

  template<BpData_t T, unsigned int DISP_VALS>
  static void printDataAndMessageValsAtPointKernelCPU(const unsigned int xVal, const unsigned int yVal,
    const beliefprop::levelProperties& currentLevelProperties,
    T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
    T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
    T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
    T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
    T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1);

  template<BpData_t T, unsigned int DISP_VALS>
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
