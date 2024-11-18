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
#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"

//include for the "kernel" functions to be run on the CPU
#include "KernelBpStereoCPU.h"

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
class ProcessOptimizedCPUBP final : public ProcessBPOnTargetDevice<T, DISP_VALS, VECTORIZATION>
{
public:
  ProcessOptimizedCPUBP(const ParallelParams& optCPUParams) : 
    ProcessBPOnTargetDevice<T, DISP_VALS, VECTORIZATION>(optCPUParams) {}

private:
  run_eval::Status initializeDataCosts(const beliefprop::BPsettings& algSettings, const beliefprop::LevelProperties& currentLevelProperties,
    const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard) override;

  run_eval::Status initializeDataCurrentLevel(const beliefprop::LevelProperties& currentLevelProperties,
    const beliefprop::LevelProperties& prevLevelProperties,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboardWriteTo,
    unsigned int bpSettingsNumDispVals) override;

  run_eval::Status initializeMessageValsToDefault(
    const beliefprop::LevelProperties& currentLevelProperties,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    unsigned int bpSettingsNumDispVals) override;

  run_eval::Status runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
    const beliefprop::LevelProperties& currentLevelProperties,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    T* allocatedMemForProcessing) override;

  run_eval::Status copyMessageValuesToNextLevelDown(
    const beliefprop::LevelProperties& currentLevelProperties,
    const beliefprop::LevelProperties& nextLevelProperties,
    const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyFrom,
    const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyTo,
    unsigned int bpSettingsNumDispVals) override;

  float* retrieveOutputDisparity(
    const beliefprop::LevelProperties& currentLevelProperties,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    unsigned int bpSettingsNumDispVals) override;
};

//functions definitions related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
  const beliefprop::CheckerboardMessages<T*>& messagesDevice,
  T* allocatedMemForProcessing)
{
  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iterationNum = 0; iterationNum < algSettings.num_iterations_; iterationNum++)
  {
    const beliefprop::Checkerboard_Part checkboardPartUpdate =
      ((iterationNum % 2) == 0) ?
      beliefprop::Checkerboard_Part::kCheckerboardPart1 :
      beliefprop::Checkerboard_Part::kCheckerboardPart0;

    beliefpropCPU::runBPIterationUsingCheckerboardUpdates<T, DISP_VALS, VECTORIZATION>(
      checkboardPartUpdate, currentLevelProperties,
      dataCostDeviceCheckerboard[0],
      dataCostDeviceCheckerboard[1],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      algSettings.disc_k_bp_, algSettings.num_disp_vals_, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::copyMessageValuesToNextLevelDown(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::LevelProperties& nextLevelProperties,
  const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyFrom,
  const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyTo,
  unsigned int bpSettingsNumDispVals)
{
  for (const auto& checkerboard_part : {beliefprop::Checkerboard_Part::kCheckerboardPart0, beliefprop::Checkerboard_Part::kCheckerboardPart1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    beliefpropCPU::copyMsgDataToNextLevel<T, DISP_VALS>(
      checkerboard_part, currentLevelProperties, nextLevelProperties,
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messagesDeviceCopyFrom[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messagesDeviceCopyTo[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      bpSettingsNumDispVals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::initializeDataCosts(
  const beliefprop::BPsettings& algSettings, const beliefprop::LevelProperties& currentLevelProperties,
  const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard)
{
  //initialize the data the the "bottom" of the image pyramid
  beliefpropCPU::initializeBottomLevelData<T, DISP_VALS>(currentLevelProperties, imagesOnTargetDevice[0],
    imagesOnTargetDevice[1], dataCostDeviceCheckerboard[0],
    dataCostDeviceCheckerboard[1], algSettings.lambda_bp_, algSettings.data_k_bp_,
    algSettings.num_disp_vals_, this->parallel_params_);
  return run_eval::Status::kNoError;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::initializeMessageValsToDefault(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::CheckerboardMessages<T*>& messagesDevice,
  unsigned int bpSettingsNumDispVals)
{
  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCPU::initializeMessageValsToDefaultKernel<T, DISP_VALS>(
    currentLevelProperties,
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    bpSettingsNumDispVals, this->parallel_params_);
  return run_eval::Status::kNoError;
}


template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::initializeDataCurrentLevel(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::LevelProperties& prevLevelProperties,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboardWriteTo,
  unsigned int bpSettingsNumDispVals)
{
  const size_t offsetNum{0};
  for (const auto& checkerboardAndDataCost : {
    std::make_pair(
      beliefprop::Checkerboard_Part::kCheckerboardPart0,
      dataCostDeviceCheckerboardWriteTo[0]),
    std::make_pair(
      beliefprop::Checkerboard_Part::kCheckerboardPart1,
      dataCostDeviceCheckerboardWriteTo[1])})
  {
    beliefpropCPU::initializeCurrentLevelData<T, DISP_VALS>(
      checkerboardAndDataCost.first, currentLevelProperties, prevLevelProperties,
      dataCostDeviceCheckerboard[0],
      dataCostDeviceCheckerboard[1],
      checkerboardAndDataCost.second,
      ((int) offsetNum / sizeof(float)),
      bpSettingsNumDispVals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline float* ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::retrieveOutputDisparity(
  const beliefprop::LevelProperties& currentLevelProperties,
  const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
  const beliefprop::CheckerboardMessages<T*>& messagesDevice,
  unsigned int bpSettingsNumDispVals)
{
  float* resultingDisparityMapCompDevice = new float[currentLevelProperties.width_level_ * currentLevelProperties.height_level_];

  beliefpropCPU::retrieveOutputDisparity<T, DISP_VALS, VECTORIZATION>(
    currentLevelProperties,
    dataCostDeviceCheckerboard[0],
    dataCostDeviceCheckerboard[1],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messagesDevice[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    resultingDisparityMapCompDevice, bpSettingsNumDispVals, this->parallel_params_);

  return resultingDisparityMapCompDevice;
}

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
