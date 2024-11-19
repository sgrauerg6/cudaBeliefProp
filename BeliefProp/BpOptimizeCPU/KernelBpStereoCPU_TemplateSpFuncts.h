#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_

//this is only processed when on x86
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "KernelBpStereoCPU.h"
#include "BpSharedFuncts/SharedBPProcessingFuncts.h"

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals,
  void* dstProcessing)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals, dstProcessing);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardToUpdate, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
    xVal, yVal, checkerboardToUpdate, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bp_settings_disp_vals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardToUpdate,
  const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  float disc_k_bp, unsigned int offsetData, bool dataAligned, unsigned int bp_settings_disp_vals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
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
void beliefprop::initializeCurrentLevelDataPixel<short, short, 0>(unsigned int xVal, unsigned int yVal,
  beliefprop::Checkerboard_Part checkerboardPart, const beliefprop::BpLevelProperties& currentBpLevel,
  const beliefprop::BpLevelProperties& prevBpLevel, short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, 0>(xVal, yVal, checkerboardPart, currentBpLevel, prevBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bp_settings_disp_vals);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, beliefprop::Checkerboard_Part checkerboardPart,
  const beliefprop::BpLevelProperties& currentBpLevel, const beliefprop::BpLevelProperties& prevBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  unsigned int offsetNum, unsigned int bp_settings_disp_vals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(xVal, yVal, checkerboardPart,
    currentBpLevel, prevBpLevel, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bp_settings_disp_vals);
}


template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, 0>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, 0>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0,
    dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[0].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[0].num_disp_vals>(xVal, yVal, currentBpLevel, dataCostStereoCheckerboard0,
    dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[1].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[1].num_disp_vals>(xVal, yVal, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[2].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[2].num_disp_vals>(xVal, yVal, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[3].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[3].num_disp_vals>(xVal, yVal, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[4].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[4].num_disp_vals>(xVal, yVal, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[5].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[5].num_disp_vals>(xVal, yVal, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::kStereoSetsToProcess[6].num_disp_vals>(
  unsigned int xVal, unsigned int yVal, const beliefprop::BpLevelProperties& currentBpLevel,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, unsigned int bp_settings_disp_vals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::kStereoSetsToProcess[6].num_disp_vals>(xVal, yVal, currentBpLevel,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bp_settings_disp_vals);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
