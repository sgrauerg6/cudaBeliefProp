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

//This function declares the host functions to run the CUDA implementation of Stereo estimation using BP

#ifndef BP_STEREO_PROCESSING_OPTIMIZED_CPU_H
#define BP_STEREO_PROCESSING_OPTIMIZED_CPU_H

#include <malloc.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpConstsAndParams/bpStructsAndEnums.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"

//include for the "kernel" functions to be run on the CPU
#include "KernelBpStereoCPU.h"

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
class ProcessOptimizedCPUBP : public ProcessBPOnTargetDevice<T, DISP_VALS, VECTORIZATION>
{
public:
  ProcessOptimizedCPUBP(const beliefprop::ParallelParameters& optCPUParams) : 
    ProcessBPOnTargetDevice<T, DISP_VALS, VECTORIZATION>(optCPUParams) {}

private:
  run_eval::Status initializeDataCosts(const beliefprop::BPsettings& algSettings, const beliefprop::levelProperties& currentLevelProperties,
    const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard) override;

  run_eval::Status initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::levelProperties& prevLevelProperties,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboardWriteTo,
    const unsigned int bpSettingsNumDispVals) override;

  run_eval::Status initializeMessageValsToDefault(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::checkerboardMessages<T*>& messagesDevice,
    const unsigned int bpSettingsNumDispVals) override;

  run_eval::Status runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
    const beliefprop::checkerboardMessages<T*>& messagesDevice,
    T* allocatedMemForProcessing) override;

  run_eval::Status copyMessageValuesToNextLevelDown(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::levelProperties& nextlevelProperties,
    const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyFrom,
    const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyTo,
    const unsigned int bpSettingsNumDispVals) override;

  float* retrieveOutputDisparity(
    const beliefprop::levelProperties& currentLevelProperties,
    const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
    const beliefprop::checkerboardMessages<T*>& messagesDevice,
    const unsigned int bpSettingsNumDispVals) override;
};

//functions definitions related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  T* allocatedMemForProcessing)
{
  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iterationNum = 0; iterationNum < algSettings.numIterations_; iterationNum++)
  {
    beliefprop::Checkerboard_Parts checkboardPartUpdate = ((iterationNum % 2) == 0) ? beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1 : beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0;

    beliefpropCPU::runBPIterationUsingCheckerboardUpdatesCPU<T, DISP_VALS, VECTORIZATION>(
      checkboardPartUpdate, currentLevelProperties,
      dataCostDeviceCheckerboard.dataCostCheckerboard0_,
      dataCostDeviceCheckerboard.dataCostCheckerboard1_,
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
      messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
      algSettings.disc_k_bp_, algSettings.numDispVals_, this->parallelParams_);
  }
  return run_eval::Status::NO_ERROR;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::copyMessageValuesToNextLevelDown(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& nextlevelProperties,
  const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyFrom,
  const beliefprop::checkerboardMessages<T*>& messagesDeviceCopyTo,
  const unsigned int bpSettingsNumDispVals)
{
  for (const auto& checkerboard_part : {beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0, beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    beliefpropCPU::copyMsgDataToNextLevelCPU<T, DISP_VALS>(
      checkerboard_part, currentLevelProperties, nextlevelProperties,
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
      messagesDeviceCopyFrom.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
      messagesDeviceCopyTo.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
      bpSettingsNumDispVals, this->parallelParams_);
  }
  return run_eval::Status::NO_ERROR;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::initializeDataCosts(
  const beliefprop::BPsettings& algSettings, const beliefprop::levelProperties& currentLevelProperties,
  const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard)
{
  //initialize the data the the "bottom" of the image pyramid
  beliefpropCPU::initializeBottomLevelDataCPU<T, DISP_VALS>(currentLevelProperties, imagesOnTargetDevice[0],
    imagesOnTargetDevice[1], dataCostDeviceCheckerboard.dataCostCheckerboard0_,
    dataCostDeviceCheckerboard.dataCostCheckerboard1_, algSettings.lambda_bp_, algSettings.data_k_bp_,
    algSettings.numDispVals_, this->parallelParams_);
  return run_eval::Status::NO_ERROR;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::initializeMessageValsToDefault(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  const unsigned int bpSettingsNumDispVals)
{
  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCPU::initializeMessageValsToDefaultKernelCPU<T, DISP_VALS>(
    currentLevelProperties,
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
    bpSettingsNumDispVals, this->parallelParams_);
  return run_eval::Status::NO_ERROR;
}


template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::initializeDataCurrentLevel(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::levelProperties& prevLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboardWriteTo,
  const unsigned int bpSettingsNumDispVals)
{
  size_t offsetNum = 0;

  for (const auto& checkerboardAndDataCost : { std::make_pair(
    beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0,
    dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0_),
    std::make_pair(beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1,
                   dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1_) })
  {
    beliefpropCPU::initializeCurrentLevelDataCPU<T, DISP_VALS>(
      checkerboardAndDataCost.first, currentLevelProperties, prevLevelProperties,
      dataCostDeviceCheckerboard.dataCostCheckerboard0_,
      dataCostDeviceCheckerboard.dataCostCheckerboard1_,
      checkerboardAndDataCost.second,
      ((int) offsetNum / sizeof(float)),
      bpSettingsNumDispVals, this->parallelParams_);
  }
  return run_eval::Status::NO_ERROR;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline float* ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::retrieveOutputDisparity(
  const beliefprop::levelProperties& currentLevelProperties,
  const beliefprop::dataCostData<T*>& dataCostDeviceCheckerboard,
  const beliefprop::checkerboardMessages<T*>& messagesDevice,
  const unsigned int bpSettingsNumDispVals)
{
  float* resultingDisparityMapCompDevice = new float[currentLevelProperties.widthLevel_ * currentLevelProperties.heightLevel_];

  beliefpropCPU::retrieveOutputDisparityCPU<T, DISP_VALS, VECTORIZATION>(
    currentLevelProperties,
    dataCostDeviceCheckerboard.dataCostCheckerboard0_,
    dataCostDeviceCheckerboard.dataCostCheckerboard1_,
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_0],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_U_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_D_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_L_CHECKERBOARD_1],
    messagesDevice.checkerboardMessagesAtLevel_[beliefprop::Message_Arrays::MESSAGES_R_CHECKERBOARD_1],
    resultingDisparityMapCompDevice, bpSettingsNumDispVals, this->parallelParams_);

  return resultingDisparityMapCompDevice;
}

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
