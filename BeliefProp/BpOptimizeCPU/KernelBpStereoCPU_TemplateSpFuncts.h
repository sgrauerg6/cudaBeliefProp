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
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, 0>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals,
  void* dstProcessing)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, 0>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals, dstProcessing);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardToUpdate, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(
    xVal, yVal, checkerboardToUpdate, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
    messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
    messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
    messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
    messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
    disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardToUpdate,
  const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
  short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
  short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
  short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
  const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
  beliefprop::runBPIterationUsingCheckerboardUpdatesKernel<short, float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(
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
void beliefprop::initializeCurrentLevelDataPixel<short, short, 0>(const unsigned int xVal, const unsigned int yVal,
  const beliefprop::Checkerboard_Parts checkerboardPart, const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
  short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, 0>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void beliefprop::initializeCurrentLevelDataPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::Checkerboard_Parts checkerboardPart,
  const beliefprop::levelProperties& currentLevelProperties, const beliefprop::levelProperties& prevLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo,
  const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
  beliefprop::initializeCurrentLevelDataPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(xVal, yVal, checkerboardPart,
    currentLevelProperties, prevLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
    offsetNum, bpSettingsDispVals);
}


template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, 0>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, 0>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
    dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
    dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>(xVal, yVal, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>(xVal, yVal, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>(xVal, yVal, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>(xVal, yVal, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>(xVal, yVal, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void beliefprop::retrieveOutputDisparityPixel<short, short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(
  const unsigned int xVal, const unsigned int yVal, const beliefprop::levelProperties& currentLevelProperties,
  short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
  short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
  short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
  beliefprop::retrieveOutputDisparityPixel<short, float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>(xVal, yVal, currentLevelProperties,
    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
    messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
    messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS_H_
