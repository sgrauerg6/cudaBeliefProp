/*
 * KernelBpStereoCPU_ARMTemplateSpFuncts.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_

#include "BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "RunImpCPU/ARMTemplateSpFuncts.h"
#include "KernelBpStereoCPU.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, 0>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, 0>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float16_t, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
  float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
  float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
  float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<float16_t, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, 0>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, 0>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<float16_t, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, 0>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, 0>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[0].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[1].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[2].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[3].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[4].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[5].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<float16_t, float16_t, bp_params::kStereoSetsToProcess[6].num_disp_vals>(unsigned int xVal, unsigned int yVal,
  const beliefprop::BpLevelProperties& currentBpLevel, float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
  float16_t* messageUPrevStereoCheckerboard0, float16_t* messageDPrevStereoCheckerboard0,
  float16_t* messageLPrevStereoCheckerboard0, float16_t* messageRPrevStereoCheckerboard0,
  float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1,
  float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1,
  float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<float16_t, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0,
    messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

#endif //COMPILING_FOR_ARM

#endif /* KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_ */
