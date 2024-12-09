/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file ProcessBpOptimizedCPU.h
 * @author Scott Grauer-Gray
 * @brief Declares the host functions to run the CUDA implementation of Stereo estimation using BP
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef PROCESS_BP_OPTIMIZED_CPU_H_
#define PROCESS_BP_OPTIMIZED_CPU_H_

#include <malloc.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include "BpRunProcessing/ProcessBp.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"

//include for the "kernel" functions to be run on the CPU
#include "KernelBpStereoCPU.h"

/**
 * @brief Class that define functions used in processing bp in the
 * optimized CPU implementation
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class ProcessBpOptimizedCPU final : public ProcessBp<T, DISP_VALS, ACCELERATION>
{
public:
  ProcessBpOptimizedCPU(const ParallelParams& opt_cpu_params) : 
    ProcessBp<T, DISP_VALS, ACCELERATION>(opt_cpu_params) {}

private:
  run_eval::Status InitializeDataCosts(
    const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const std::array<float*, 2>& images_target_device,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device) const override;

  run_eval::Status InitializeDataCurrentLevel(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& prev_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
    unsigned int bp_settings_num_disp_vals) const override;

  run_eval::Status InitializeMessageValsToDefault(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) const override;

  run_eval::Status RunBPAtCurrentLevel(
    const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    T* allocated_memory) const override;

  run_eval::Status CopyMessageValuesToNextLevelDown(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& next_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
    unsigned int bp_settings_num_disp_vals) const override;

  float* RetrieveOutputDisparity(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) const override;
};

//functions definitions related to running BP to retrieve the movement between the images

//run the given number of iterations of BP at the current level using the given message values in global device memory
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>::RunBPAtCurrentLevel(
  const beliefprop::BpSettings& alg_settings,
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  T* allocated_memory) const
{
  //at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
  for (unsigned int iteration_num = 0; iteration_num < alg_settings.num_iterations; iteration_num++)
  {
    const beliefprop::CheckerboardPart checkerboard_part_update =
      ((iteration_num % 2) == 0) ?
      beliefprop::CheckerboardPart::kCheckerboardPart1 :
      beliefprop::CheckerboardPart::kCheckerboardPart0;

    using namespace beliefprop;
    beliefpropCPU::RunBPIterationUsingCheckerboardUpdates<T, DISP_VALS, ACCELERATION>(
      checkerboard_part_update, current_bp_level.LevelProperties(),
      data_costs_device[0], data_costs_device[1],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
      messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
      alg_settings.disc_k_bp, alg_settings.num_disp_vals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>::CopyMessageValuesToNextLevelDown(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::BpLevel& next_bp_level,
  const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
  const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
  unsigned int bp_settings_num_disp_vals) const
{
  for (const auto& checkerboard_part : {beliefprop::CheckerboardPart::kCheckerboardPart0,
                                        beliefprop::CheckerboardPart::kCheckerboardPart1})
  {
    //call the kernel to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
    //storing the current message values
    using namespace beliefprop;
    beliefpropCPU::CopyMsgDataToNextLevel<T, DISP_VALS>(
      checkerboard_part, current_bp_level.LevelProperties(), next_bp_level.LevelProperties(),
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
      messages_device_copy_from[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
      messages_device_copy_to[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
      bp_settings_num_disp_vals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>::InitializeDataCosts(
  const beliefprop::BpSettings& alg_settings, const beliefprop::BpLevel& current_bp_level,
  const std::array<float*, 2>& images_target_device, const beliefprop::DataCostsCheckerboards<T*>& data_costs_device) const
{
  //initialize the data the the "bottom" of the image pyramid
  beliefpropCPU::InitializeBottomLevelData<T, DISP_VALS>(
    current_bp_level.LevelProperties(), 
    images_target_device[0],images_target_device[1],
    data_costs_device[0], data_costs_device[1],
    alg_settings.lambda_bp, alg_settings.data_k_bp,
    alg_settings.num_disp_vals, this->parallel_params_);
  return run_eval::Status::kNoError;
}

//initialize the message values with no previous message values...all message values are set to 0
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>::InitializeMessageValsToDefault(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  unsigned int bp_settings_num_disp_vals) const
{
  using namespace beliefprop;
  //initialize all the message values for each pixel at each possible movement to the default value in the kernel
  beliefpropCPU::InitializeMessageValsToDefaultKernel<T, DISP_VALS>(
    current_bp_level.LevelProperties(),
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
    bp_settings_num_disp_vals, this->parallel_params_);

  return run_eval::Status::kNoError;
}


template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline run_eval::Status ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>::InitializeDataCurrentLevel(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::BpLevel& prev_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
  unsigned int bp_settings_num_disp_vals) const
{
  const size_t offset_num{0};
  for (const auto& checkerboard_data_cost : {
    std::make_pair(
      beliefprop::CheckerboardPart::kCheckerboardPart0,
      data_costs_device_write[0]),
    std::make_pair(
      beliefprop::CheckerboardPart::kCheckerboardPart1,
      data_costs_device_write[1])})
  {
    beliefpropCPU::InitializeCurrentLevelData<T, DISP_VALS>(
      checkerboard_data_cost.first,
      current_bp_level.LevelProperties(), prev_bp_level.LevelProperties(),
      data_costs_device[0], data_costs_device[1],
      checkerboard_data_cost.second,
      ((int) offset_num / sizeof(float)),
      bp_settings_num_disp_vals, this->parallel_params_);
  }
  return run_eval::Status::kNoError;
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline float* ProcessBpOptimizedCPU<T, DISP_VALS, ACCELERATION>::RetrieveOutputDisparity(
  const beliefprop::BpLevel& current_bp_level,
  const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
  const beliefprop::CheckerboardMessages<T*>& messages_device,
  unsigned int bp_settings_num_disp_vals) const
{
  float* result_disp_map_device = 
    new float[current_bp_level.LevelProperties().width_level_ * current_bp_level.LevelProperties().height_level_];

  using namespace beliefprop;
  beliefpropCPU::RetrieveOutputDisparity<T, DISP_VALS, ACCELERATION>(
    current_bp_level.LevelProperties(),
    data_costs_device[0],
    data_costs_device[1],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart0)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesUCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesDCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesLCheckerboard)],
    messages_device[static_cast<unsigned int>(CheckerboardPart::kCheckerboardPart1)][static_cast<unsigned int>(MessageArrays::kMessagesRCheckerboard)],
    result_disp_map_device, bp_settings_num_disp_vals, this->parallel_params_);

  return result_disp_map_device;
}

#endif //PROCESS_BP_OPTIMIZED_CPU_H_
