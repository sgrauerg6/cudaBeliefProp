/*
 * KernelBpStereoCPU_ARMTemplateSpFuncts.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_

#include "KernelBpStereoCPU.h"
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

template<> inline
float16_t getZeroVal<float16_t>()
{
  return (float16_t)0.0f;
}

template<> inline
float convertValToDifferentDataTypeIfNeeded<float16_t, float>(const float16_t valToConvert)
{
  return (float)valToConvert;
}

template<> inline
float16_t convertValToDifferentDataTypeIfNeeded<float, float16_t>(const float valToConvert)
{
  //seems like simple cast function works
  return (float16_t)valToConvert;
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, 0>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, 0>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, 0>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, 0>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  initializeCurrentLevelDataStereoPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, 0>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, 0>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

#endif //COMPILING_FOR_ARM

#endif /* KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_ */
