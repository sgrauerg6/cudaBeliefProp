/*
 * ProcessBPOnTargetDevice.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICE_H_
#define PROCESSBPONTARGETDEVICE_H_

#include <math.h>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <ranges>
#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "RunImp/RunImpMemoryManagement.h"
#include "BpRunSettings.h"
#include "BpLevel.h"
#include "BpSettings.h"
#include "BpParallelParams.h"

//alias for time point for start and end time for each timing segment
using timingType = std::chrono::time_point<std::chrono::system_clock>;

//Abstract class to process belief propagation on target device
//Some of the class functions need to be overridden to for processing on
//target device
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class ProcessBPOnTargetDevice {
public:
  ProcessBPOnTargetDevice(const ParallelParams& parallel_params) : parallel_params_{parallel_params} { }

  virtual run_eval::Status ErrorCheck(const char *file = "", int line = 0, bool abort = false) const {
    return run_eval::Status::kNoError;
  }
  
  //run the belief propagation algorithm with on a set of stereo images to generate a disparity map
  //input is images image1Pixels and image1Pixels
  //output is resultingDisparityMap
  std::optional<std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>> operator()(const std::array<float*, 2>& images_target_device,
    const beliefprop::BpSettings& alg_settings, const std::array<unsigned int, 2>& width_height_images,
    T* allocatedMemForBpProcessingDevice, T* allocated_memory,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run);

protected:
  const ParallelParams& parallel_params_;

private:
  //initialize data cost for each possible disparity at bottom level
  virtual run_eval::Status InitializeDataCosts(const beliefprop::BpSettings& alg_settings, const beliefprop::BpLevel& current_bp_level,
    const std::array<float*, 2>& images_target_device, const beliefprop::DataCostsCheckerboards<T*>& data_costs_device) = 0;

  //initialize data cost for each possible disparity at levels above the bottom level
  virtual run_eval::Status InitializeDataCurrentLevel(const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& prev_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device_write,
    unsigned int bp_settings_num_disp_vals) = 0;

  //initialize message values at first level of bp processing to default value
  virtual run_eval::Status InitializeMessageValsToDefault(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) = 0;

  //run belief propagation processing at current level of processing hierarchy
  virtual run_eval::Status RunBPAtCurrentLevel(const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    T* allocated_memory) = 0;

  //copy message values from current level of processing to next level of processing
  virtual run_eval::Status CopyMessageValuesToNextLevelDown(
    const beliefprop::BpLevel& current_bp_level,
    const beliefprop::BpLevel& next_bp_level,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_from,
    const beliefprop::CheckerboardMessages<T*>& messages_device_copy_to,
    unsigned int bp_settings_num_disp_vals) = 0;

  //retrieve computed output disparity at each pixel in bottom level using data and message values
  virtual float* RetrieveOutputDisparity(
    const beliefprop::BpLevel& BpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_device,
    const beliefprop::CheckerboardMessages<T*>& messages_device,
    unsigned int bp_settings_num_disp_vals) = 0;

  //free memory used for message values in bp processing
  virtual void FreeCheckerboardMessagesMemory(
    const beliefprop::CheckerboardMessages<T*>& checkerboard_messages_to_free,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    std::ranges::for_each(checkerboard_messages_to_free,
      [this, &mem_management_bp_run](auto& checkerboardMessagesSet) {
      mem_management_bp_run->FreeAlignedMemoryOnDevice(checkerboardMessagesSet); });
  }

  //allocate memory for message values in bp processing
  virtual beliefprop::CheckerboardMessages<T*> AllocateMemoryForCheckerboardMessages(
    const unsigned long num_data_allocate_per_message,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    beliefprop::CheckerboardMessages<T*> output_checkerboard_messages;
    std::ranges::for_each(output_checkerboard_messages,
      [this, num_data_allocate_per_message, &mem_management_bp_run](auto& checkerboardMessagesSet) {
      checkerboardMessagesSet =
        mem_management_bp_run->AllocateAlignedMemoryOnDevice(num_data_allocate_per_message, ACCELERATION); });

    return output_checkerboard_messages;
  }

  //retrieve pointer to bp message data at current level using specified offset
  virtual beliefprop::CheckerboardMessages<T*> RetrieveLevelMessageData(
    const beliefprop::CheckerboardMessages<T*>& all_checkerboard_messages,
    const unsigned long offset_into_messages)
  {
    beliefprop::CheckerboardMessages<T*> output_checkerboard_messages;
    for (unsigned int i = 0; i < output_checkerboard_messages.size(); i++) {
      output_checkerboard_messages[i] =
        &((all_checkerboard_messages[i])[offset_into_messages]);
    }

    return output_checkerboard_messages;
  }

  //free memory allocated for data costs in bp processing
  virtual void FreeDataCostsMemory(const beliefprop::DataCostsCheckerboards<T*>& data_costs_to_free,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run) {
    mem_management_bp_run->FreeAlignedMemoryOnDevice(data_costs_to_free[0]);
    mem_management_bp_run->FreeAlignedMemoryOnDevice(data_costs_to_free[1]);
  }

  //allocate memory for data costs in bp processing
  virtual beliefprop::DataCostsCheckerboards<T*> AllocateMemoryForDataCosts(
    const unsigned long num_data_costs_checkerboards,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run) {
    return {mem_management_bp_run->AllocateAlignedMemoryOnDevice(num_data_costs_checkerboards, ACCELERATION), 
            mem_management_bp_run->AllocateAlignedMemoryOnDevice(num_data_costs_checkerboards, ACCELERATION)};
  }

  //allocate and organize data cost and message value data at all levels for bp processing
  virtual std::pair<beliefprop::DataCostsCheckerboards<T*>, beliefprop::CheckerboardMessages<T*>>
    AllocateAndOrganizeDataCostsAndMessageDataAllLevels(
      const unsigned long num_data_allocate_per_data_costs_message_data_array,
      const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    T* data_all_levels = mem_management_bp_run->AllocateAlignedMemoryOnDevice(
      10u*num_data_allocate_per_data_costs_message_data_array, ACCELERATION);
    return OrganizeDataCostsAndMessageDataAllLevels(
      data_all_levels, num_data_allocate_per_data_costs_message_data_array);
  }

  //organize data cost and message value data at all bp processing levels
  virtual std::pair<beliefprop::DataCostsCheckerboards<T*>, beliefprop::CheckerboardMessages<T*>>
    OrganizeDataCostsAndMessageDataAllLevels(
      T* data_all_levels, const unsigned long num_data_allocate_per_data_costs_message_data_array)
  {
    beliefprop::DataCostsCheckerboards<T*> data_costs_device_checkerboard_all_levels;
    data_costs_device_checkerboard_all_levels[0] = data_all_levels;
    data_costs_device_checkerboard_all_levels[1] =
      &(data_costs_device_checkerboard_all_levels[0][1 * (num_data_allocate_per_data_costs_message_data_array)]);

    beliefprop::CheckerboardMessages<T*> messages_device_all_levels;
    for (unsigned int i = 0; i < messages_device_all_levels.size(); i++) {
      messages_device_all_levels[i] =
        &(data_costs_device_checkerboard_all_levels[0][(i + 2) * (num_data_allocate_per_data_costs_message_data_array)]);
    }

    return {data_costs_device_checkerboard_all_levels, messages_device_all_levels};
  }

  //free data costs at all levels for bp processing that are all together in a single array
  virtual void FreeDataCostsAllDataInSingleArray(
    const beliefprop::DataCostsCheckerboards<T*>& data_costs_to_free,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    mem_management_bp_run->FreeAlignedMemoryOnDevice(data_costs_to_free[0]);
  }

  //retrieve pointer to data costs for level using specified offset
  virtual beliefprop::DataCostsCheckerboards<T*> RetrieveLevelDataCosts(
    const beliefprop::DataCostsCheckerboards<T*>& all_data_costs,
    const unsigned long offset_into_all_data_costs)
  {
    return {&(all_data_costs[0][offset_into_all_data_costs]),
            &(all_data_costs[1][offset_into_all_data_costs])};
  }
};

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map on target device
//input is images on target device for computation
//output is disparity map and processing runtimes
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
std::optional<std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>>
  ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>::operator()(
    const std::array<float*, 2> & images_target_device, const beliefprop::BpSettings& alg_settings,
    const std::array<unsigned int, 2>& width_height_images, T* allocatedMemForBpProcessingDevice, T* allocated_memory,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
{
  if (ErrorCheck() != run_eval::Status::kNoError) { return {}; }

  std::unordered_map<beliefprop::Runtime_Type, std::array<timingType, 2>> start_end_times;
  std::vector<std::array<timingType, 2>> data_costs_timings(alg_settings.num_levels);
  std::vector<std::array<timingType, 2>> bp_timings(alg_settings.num_levels);
  std::vector<std::array<timingType, 2>> data_copy_timings(alg_settings.num_levels);
  std::chrono::duration<double> total_time_bp_iters{0}, total_time_copy_data{0}, total_time_copy_data_kernel{0};

  //start at the "bottom level" and work way up to determine amount of space needed to store data costs
  std::vector<beliefprop::BpLevel> bp_levels;
  bp_levels.reserve(alg_settings.num_levels);

  //set level properties for bottom level that include processing of full image width/height
  bp_levels.push_back(beliefprop::BpLevel(width_height_images, 0, 0, ACCELERATION));

  //compute level properties which includes offset for each data/message array for each level after the bottom level
  for (unsigned int level_num = 1; level_num < alg_settings.num_levels; level_num++) {
    //get current level properties from previous level properties
    bp_levels.push_back(bp_levels[level_num-1].NextBpLevel<T>(alg_settings.num_disp_vals));
  }

  start_end_times[beliefprop::Runtime_Type::kInitSettingsMalloc][0] = std::chrono::system_clock::now();

  //declare and allocate the space on the device to store the data cost component at each
  //possible movement at each level of the "pyramid" as well as the message data used for bp processing
  //each checkerboard holds half of the data and checkerboard 0 includes the pixel in slot (0, 0)
  beliefprop::DataCostsCheckerboards<T*> data_costs_device_all_levels;
  beliefprop::CheckerboardMessages<T*> messages_device_all_levels;

  //data for each array at all levels set to data up to final level (corresponds to offset at final level) plus data amount at final level
  const unsigned long data_all_levels_each_data_message_arr =
    bp_levels[alg_settings.num_levels-1].LevelProperties().offset_into_arrays_ +
    bp_levels[alg_settings.num_levels-1].NumDataInBpArrays<T>(alg_settings.num_disp_vals);

  //assuming that width includes padding
  if constexpr (beliefprop::kUseOptGPUMemManagement) {
    if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
      std::tie(data_costs_device_all_levels, messages_device_all_levels) =
        OrganizeDataCostsAndMessageDataAllLevels(
          allocatedMemForBpProcessingDevice, data_all_levels_each_data_message_arr);
    }
    else {
      //call function that allocates all data in single array and then set offsets in array for data costs and message data locations
      std::tie(data_costs_device_all_levels, messages_device_all_levels) =
        AllocateAndOrganizeDataCostsAndMessageDataAllLevels(
          data_all_levels_each_data_message_arr, mem_management_bp_run);
    }
  }
  else {
    data_costs_device_all_levels = AllocateMemoryForDataCosts(data_all_levels_each_data_message_arr, mem_management_bp_run);
  }

  auto curr_time = std::chrono::system_clock::now();
  start_end_times[beliefprop::Runtime_Type::kInitSettingsMalloc][1] = curr_time;
  data_costs_timings[0][0] = curr_time;

  //initialize the data cost at the bottom level
  auto error_code = InitializeDataCosts(alg_settings, bp_levels[0], images_target_device, data_costs_device_all_levels);
  if (error_code != run_eval::Status::kNoError) { return {}; }

  curr_time = std::chrono::system_clock::now();
  data_costs_timings[0][1] = curr_time;
  start_end_times[beliefprop::Runtime_Type::kDataCostsHigherLevel][0] = curr_time;

  //set the data costs at each level from the bottom level "up"
  for (unsigned int level_num = 1u; level_num < alg_settings.num_levels; level_num++)
  {
    data_costs_timings[level_num][0] = std::chrono::system_clock::now();
    error_code = InitializeDataCurrentLevel(bp_levels[level_num], bp_levels[level_num - 1],
      RetrieveLevelDataCosts(data_costs_device_all_levels, bp_levels[level_num - 1u].LevelProperties().offset_into_arrays_),
      RetrieveLevelDataCosts(data_costs_device_all_levels, bp_levels[level_num].LevelProperties().offset_into_arrays_),
      alg_settings.num_disp_vals);
    if (error_code != run_eval::Status::kNoError) { return {}; }

    data_costs_timings[level_num][1] = std::chrono::system_clock::now();
  }

  curr_time = data_costs_timings[alg_settings.num_levels-1][1];
  start_end_times[beliefprop::Runtime_Type::kDataCostsHigherLevel][1] = curr_time;
  start_end_times[beliefprop::Runtime_Type::kInitMessages][0] = curr_time;

  //get and use offset into data at current processing level of pyramid
  beliefprop::DataCostsCheckerboards<T*> data_costs_device_current_level = RetrieveLevelDataCosts(
    data_costs_device_all_levels, bp_levels[alg_settings.num_levels - 1u].LevelProperties().offset_into_arrays_);

  //declare the space to pass the BP messages; need to have two "sets" of checkerboards because the message
  //data at the "higher" level in the image pyramid need copied to a lower level without overwriting values
  std::array<beliefprop::CheckerboardMessages<T*>, 2> messages_device;

  //assuming that width includes padding
  if constexpr (beliefprop::kUseOptGPUMemManagement) {
    messages_device[0] = RetrieveLevelMessageData(
      messages_device_all_levels, bp_levels[alg_settings.num_levels - 1u].LevelProperties().offset_into_arrays_);
  }
  else {
    //allocate the space for the message values in the first checkboard set at the current level
    messages_device[0] = AllocateMemoryForCheckerboardMessages(
      bp_levels[alg_settings.num_levels - 1u].NumDataInBpArrays<T>(alg_settings.num_disp_vals), mem_management_bp_run);
  }

  start_end_times[beliefprop::Runtime_Type::kInitMessagesKernel][0] = std::chrono::system_clock::now();

  //initialize all the BP message values at every pixel for every disparity to 0
  error_code = InitializeMessageValsToDefault(
    bp_levels[alg_settings.num_levels - 1u], messages_device[0], alg_settings.num_disp_vals);
  if (error_code != run_eval::Status::kNoError) { return {}; }

  curr_time = std::chrono::system_clock::now();
  start_end_times[beliefprop::Runtime_Type::kInitMessagesKernel][1] = curr_time;
  start_end_times[beliefprop::Runtime_Type::kInitMessages][1] = curr_time;

  //alternate between checkerboard sets 0 and 1
  enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
  Checkerboard_Num current_checkerboard_set{Checkerboard_Num::CHECKERBOARD_ZERO};

  //run BP at each level in the "pyramid" starting on top and continuing to the bottom
  //where the final movement values are computed...the message values are passed from
  //the upper level to the lower levels; this pyramid methods causes the BP message values
  //to converge more quickly
  for (int level_num = (int)alg_settings.num_levels - 1; level_num >= 0; level_num--)
  {
    const auto bp_iter_start_time = std::chrono::system_clock::now();

    //alternate which checkerboard set to work on since copying from one to the other
    //to avoid read-write conflict when copying in parallel
    error_code = RunBPAtCurrentLevel(
      alg_settings, bp_levels[(unsigned int)level_num],
      data_costs_device_current_level,
      messages_device[current_checkerboard_set],
      allocated_memory);
    if (error_code != run_eval::Status::kNoError) { return {}; }

    const auto bp_iter_end_time = std::chrono::system_clock::now();
    total_time_bp_iters += bp_iter_end_time - bp_iter_start_time;
    const auto copy_message_values_start_time = std::chrono::system_clock::now();

    //if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
    if (level_num > 0)
    {
      //use offset into allocated memory at next level
      data_costs_device_current_level = 
        RetrieveLevelDataCosts(data_costs_device_all_levels, bp_levels[level_num - 1].LevelProperties().offset_into_arrays_);

      //assuming that width includes padding
      if constexpr (beliefprop::kUseOptGPUMemManagement) {
        messages_device[(current_checkerboard_set + 1) % 2] = RetrieveLevelMessageData(
          messages_device_all_levels, bp_levels[level_num - 1].LevelProperties().offset_into_arrays_);
      }
      else {
        //allocate space in the GPU for the message values in the checkerboard set to copy to
        messages_device[(current_checkerboard_set + 1) % 2] = AllocateMemoryForCheckerboardMessages(
          bp_levels[level_num - 1].NumDataInBpArrays<T>(alg_settings.num_disp_vals));
      }

      const auto copy_message_values_kernel_start_time = std::chrono::system_clock::now();

      //currentCheckerboardSet = index copying data from
      //(currentCheckerboardSet + 1) % 2 = index copying data to
      error_code = CopyMessageValuesToNextLevelDown(bp_levels[level_num], bp_levels[level_num - 1],
        messages_device[current_checkerboard_set], messages_device[(current_checkerboard_set + 1) % 2],
        alg_settings.num_disp_vals);
      if (error_code != run_eval::Status::kNoError) { return {}; }

      const auto copy_message_values_kernel_end_time = std::chrono::system_clock::now();
      total_time_copy_data_kernel += copy_message_values_kernel_end_time - copy_message_values_kernel_start_time;
      data_copy_timings[level_num][0] = copy_message_values_kernel_start_time;
      data_copy_timings[level_num][1] = copy_message_values_kernel_end_time;

      //assuming that width includes padding
      if constexpr (!beliefprop::kUseOptGPUMemManagement) {
        //free the now-copied from computed data of the completed level
        FreeCheckerboardMessagesMemory(messages_device[current_checkerboard_set], mem_management_bp_run);
      }

      //alternate between checkerboard parts 1 and 2
      current_checkerboard_set = (current_checkerboard_set == Checkerboard_Num::CHECKERBOARD_ZERO) ?
        Checkerboard_Num::CHECKERBOARD_ONE : Checkerboard_Num::CHECKERBOARD_ZERO;
    }

    const auto copy_message_values_end_time = std::chrono::system_clock::now();
    total_time_copy_data += copy_message_values_end_time - copy_message_values_start_time;
    bp_timings[level_num][0] = bp_iter_start_time;
    bp_timings[level_num][1] = bp_iter_end_time;
  }

  start_end_times[beliefprop::Runtime_Type::kOutputDisparity][0] = std::chrono::system_clock::now();

  //assume in bottom level when retrieving output disparity
  float* result_disp_map_device = RetrieveOutputDisparity(bp_levels[0],
    data_costs_device_current_level, messages_device[current_checkerboard_set], alg_settings.num_disp_vals);
  if (result_disp_map_device == nullptr) { return {}; }

  curr_time = std::chrono::system_clock::now();
  start_end_times[beliefprop::Runtime_Type::kOutputDisparity][1] = curr_time;
  start_end_times[beliefprop::Runtime_Type::kFinalFree][0] = curr_time;

  if constexpr (beliefprop::kUseOptGPUMemManagement) {
    if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
      //do nothing; memory free outside of runs
    }
    else {
      //now free the allocated data space; all data in single array when
      //beliefprop::kUseOptGPUMemManagement set to true
      FreeDataCostsAllDataInSingleArray(data_costs_device_all_levels, mem_management_bp_run);
    }
  }
  else {
    //free the device storage allocated to the message values used to retrieve the output movement values
    FreeCheckerboardMessagesMemory(messages_device[(current_checkerboard_set == 0) ? 0 : 1], mem_management_bp_run);

    //now free the allocated data space
    FreeDataCostsMemory(data_costs_device_all_levels, mem_management_bp_run);
  }

  start_end_times[beliefprop::Runtime_Type::kFinalFree][1] = std::chrono::system_clock::now();

  start_end_times[beliefprop::Runtime_Type::kLevel0DataCosts] = data_costs_timings[0];
  start_end_times[beliefprop::Runtime_Type::kLevel0Bp] = bp_timings[0];
  start_end_times[beliefprop::Runtime_Type::kLevel0Copy] = data_copy_timings[0];
  if (bp_timings.size() > 1) {
    start_end_times[beliefprop::Runtime_Type::kLevel1DataCosts] = data_costs_timings[1];
    start_end_times[beliefprop::Runtime_Type::kLevel1Bp] = bp_timings[1];
    start_end_times[beliefprop::Runtime_Type::kLevel1Copy] = data_copy_timings[1];
  }
  if (bp_timings.size() > 2) {
    start_end_times[beliefprop::Runtime_Type::kLevel2DataCosts] = data_costs_timings[2];
    start_end_times[beliefprop::Runtime_Type::kLevel2Bp] = bp_timings[2];
    start_end_times[beliefprop::Runtime_Type::kLevel2Copy] = data_copy_timings[2];
  }
  if (bp_timings.size() > 3) {
    start_end_times[beliefprop::Runtime_Type::kLevel3DataCosts] = data_costs_timings[3];
    start_end_times[beliefprop::Runtime_Type::kLevel3Bp] = bp_timings[3];
    start_end_times[beliefprop::Runtime_Type::kLevel3Copy] = data_copy_timings[3];
  }
  if (bp_timings.size() > 4) {
    start_end_times[beliefprop::Runtime_Type::kLevel4DataCosts] = data_costs_timings[4];
    start_end_times[beliefprop::Runtime_Type::kLevel4Bp] = bp_timings[4];
    start_end_times[beliefprop::Runtime_Type::kLevel4Copy] = data_copy_timings[4];
  }
  if (bp_timings.size() > 5) {
    start_end_times[beliefprop::Runtime_Type::kLevel5DataCosts] = data_costs_timings[5];
    start_end_times[beliefprop::Runtime_Type::kLevel5Bp] = bp_timings[5];
    start_end_times[beliefprop::Runtime_Type::kLevel5Copy] = data_copy_timings[5];
  }
  if (bp_timings.size() > 6) {
    start_end_times[beliefprop::Runtime_Type::kLevel6DataCosts] = data_costs_timings[6];
    start_end_times[beliefprop::Runtime_Type::kLevel6Bp] = bp_timings[6];
    start_end_times[beliefprop::Runtime_Type::kLevel6Copy] = data_copy_timings[6];
  }
  if (bp_timings.size() > 7) {
    start_end_times[beliefprop::Runtime_Type::kLevel7DataCosts] = data_costs_timings[7];
    start_end_times[beliefprop::Runtime_Type::kLevel7Bp] = bp_timings[7];
    start_end_times[beliefprop::Runtime_Type::kLevel7Copy] = data_copy_timings[7];
  }
  if (bp_timings.size() > 8) {
    start_end_times[beliefprop::Runtime_Type::kLevel8DataCosts] = data_costs_timings[8];
    start_end_times[beliefprop::Runtime_Type::kLevel8Bp] = bp_timings[8];
    start_end_times[beliefprop::Runtime_Type::kLevel8Copy] = data_copy_timings[8];
  }
  if (bp_timings.size() > 9) {
    start_end_times[beliefprop::Runtime_Type::kLevel9DataCosts] = data_costs_timings[9];
    start_end_times[beliefprop::Runtime_Type::kLevel9Bp] = bp_timings[9];
    start_end_times[beliefprop::Runtime_Type::kLevel9Copy] = data_copy_timings[9];
  }

  //add timing for each runtime segment to segment_timings object
  DetailedTimings segment_timings(beliefprop::kTimingNames);
  std::ranges::for_each(start_end_times,
    [&segment_timings](const auto& current_runtime_name_timing) {
    segment_timings.AddTiming(current_runtime_name_timing.first,
      current_runtime_name_timing.second[1] - current_runtime_name_timing.second[0]);
  });

  segment_timings.AddTiming(beliefprop::Runtime_Type::kBpIters, total_time_bp_iters);
  segment_timings.AddTiming(beliefprop::Runtime_Type::kCopyData, total_time_copy_data);
  segment_timings.AddTiming(beliefprop::Runtime_Type::kCopyDataKernel, total_time_copy_data_kernel);

  const auto total_time = segment_timings.MedianTiming(beliefprop::Runtime_Type::kInitSettingsMalloc)
    + segment_timings.MedianTiming(beliefprop::Runtime_Type::kLevel0DataCosts)
    + segment_timings.MedianTiming(beliefprop::Runtime_Type::kDataCostsHigherLevel)
    + segment_timings.MedianTiming(beliefprop::Runtime_Type::kInitMessages)
    + total_time_bp_iters
    + total_time_copy_data + segment_timings.MedianTiming(beliefprop::Runtime_Type::kOutputDisparity)
    + segment_timings.MedianTiming(beliefprop::Runtime_Type::kFinalFree);
  segment_timings.AddTiming(beliefprop::Runtime_Type::kTotalTimed, total_time);

  return std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>{result_disp_map_device, segment_timings};
}

#endif /* PROCESSBPONTARGETDEVICE_H_ */
