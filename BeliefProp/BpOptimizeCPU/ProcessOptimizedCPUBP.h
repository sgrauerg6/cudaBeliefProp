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
  ProcessOptimizedCPUBP(const ParallelParams& opt_cpu_params) : 
    ProcessBPOnTargetDevice<T, DISP_VALS, VECTORIZATION>(opt_cpu_params) {}

private:
  run_eval::Status InitializeDataCosts(const beliefprop::BpSettings& alg_settings, const beliefprop::BpLevel& current_bp_level,
    const std::array<float*, 2>& images_target_device, const beliefprop::DataCostsCheckerboards<T*>& data_costs_device) override;

  run_eval::Status InitializeDataCurrentLevel(const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& prev_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
    unsigned int bp_settings_num_disp_vals) override;

  run_eval::Status InitializeMessageValsToDefault(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) override;

  run_eval::Status RunBPAtCurrentLevel(const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    T* allocated_memory) override;

  run_eval::Status CopyMessageValuesToNextLevelDown(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& next_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
    unsigned int bp_settings_num_disp_vals) override;

  float* RetrieveOutputDisparity(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) override;
};

//functions definitions related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::RunBPAtCurrentLevel(const beliefprop::BpSettings& alg_settings,
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  T* allocated_memory)
{
  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iteration_num = 0; iteration_num < alg_settings.num_iterations; iteration_num++)
  {
    const beliefprop::Checkerboard_Part checkerboard_part_update =
      ((iteration_num % 2) == 0) ?
      beliefprop::Checkerboard_Part::kCheckerboardPart1 :
      beliefprop::Checkerboard_Part::kCheckerboardPart0;

    beliefpropCPU::RunBPIterationUsingCheckerboardUpdates<T, DISP_VALS, VECTORIZATION>(
      checkerboard_part_update, current_bp_level.LevelProperties(),
      data_costs_device[0],
      data_costs_device[1],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      alg_settings.disc_k_bp, alg_settings.num_disp_vals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::CopyMessageValuesToNextLevelDown(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::BpLevel& next_bp_level,
  const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
  const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
  unsigned int bp_settings_num_disp_vals)
{
  for (const auto& checkerboard_part : {beliefprop::Checkerboard_Part::kCheckerboardPart0, beliefprop::Checkerboard_Part::kCheckerboardPart1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    beliefpropCPU::CopyMsgDataToNextLevel<T, DISP_VALS>(
      checkerboard_part, current_bp_level.LevelProperties(), next_bp_level.LevelProperties(),
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messages_device_copy_from[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
      messages_device_copy_to[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
      bp_settings_num_disp_vals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::InitializeDataCosts(
  const beliefprop::BpSettings& alg_settings, const beliefprop::BpLevel& current_bp_level,
  const std::array<float*, 2>& images_target_device, const beliefprop::DataCostsCheckerboards<T*>& data_costs_device)
{
  //initialize the data the the "bottom" of the image pyramid
  beliefpropCPU::InitializeBottomLevelData<T, DISP_VALS>(current_bp_level.LevelProperties(), images_target_device[0],
    images_target_device[1], data_costs_device[0],
    data_costs_device[1], alg_settings.lambda_bp, alg_settings.data_k_bp,
    alg_settings.num_disp_vals, this->parallel_params_);
  return run_eval::Status::kNoError;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::InitializeMessageValsToDefault(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  unsigned int bp_settings_num_disp_vals)
{
  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCPU::InitializeMessageValsToDefaultKernel<T, DISP_VALS>(
    current_bp_level.LevelProperties(),
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    bp_settings_num_disp_vals, this->parallel_params_);
  return run_eval::Status::kNoError;
}


template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline run_eval::Status ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::InitializeDataCurrentLevel(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::BpLevel& prev_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
  unsigned int bp_settings_num_disp_vals)
{
  const size_t offset_num{0};
  for (const auto& checkerboard_data_cost : {
    std::make_pair(
      beliefprop::Checkerboard_Part::kCheckerboardPart0,
      data_costs_device_write[0]),
    std::make_pair(
      beliefprop::Checkerboard_Part::kCheckerboardPart1,
      data_costs_device_write[1])})
  {
    beliefpropCPU::InitializeCurrentLevelData<T, DISP_VALS>(
      checkerboard_data_cost.first, current_bp_level.LevelProperties(), prev_bp_level.LevelProperties(),
      data_costs_device[0],
      data_costs_device[1],
      checkerboard_data_cost.second,
      ((int) offset_num / sizeof(float)),
      bp_settings_num_disp_vals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline float* ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>::RetrieveOutputDisparity(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  unsigned int bp_settings_num_disp_vals)
{
  float* result_disp_map_device = new float[current_bp_level.LevelProperties().width_level_ * current_bp_level.LevelProperties().height_level_];

  beliefpropCPU::RetrieveOutputDisparity<T, DISP_VALS, VECTORIZATION>(
    current_bp_level.LevelProperties(),
    data_costs_device[0],
    data_costs_device[1],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard0)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesUCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesDCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesLCheckerboard1)],
    messages_device[static_cast<unsigned int>(beliefprop::Message_Arrays::kMessagesRCheckerboard1)],
    result_disp_map_device, bp_settings_num_disp_vals, this->parallel_params_);

  return result_disp_map_device;
}

#endif //RUN_BP_STEREO_HOST_HEADER_CUH
