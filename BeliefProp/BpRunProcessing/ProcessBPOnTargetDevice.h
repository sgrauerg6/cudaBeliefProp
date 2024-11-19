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
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "BpRunImp/BpLevel.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "RunImp/RunImpMemoryManagement.h"

//alias for time point for start and end time for each timing segment
using timingType = std::chrono::time_point<std::chrono::system_clock>;

//Abstract class to process belief propagation on target device
//Some of the class functions need to be overridden to for processing on
//target device
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class ProcessBPOnTargetDevice {
public:
  ProcessBPOnTargetDevice(const ParallelParams& parallel_params) : parallel_params_{parallel_params} { }

  virtual run_eval::Status errorCheck(const char *file = "", int line = 0, bool abort = false) const {
    return run_eval::Status::kNoError;
  }
  
  //run the belief propagation algorithm with on a set of stereo images to generate a disparity map
  //input is images image1Pixels and image1Pixels
  //output is resultingDisparityMap
  std::optional<std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>> operator()(const std::array<float*, 2>& imagesOnTargetDevice,
    const beliefprop::BpSettings& algSettings, const std::array<unsigned int, 2>& width_heightImages,
    T* allocatedMemForBpProcessingDevice, T* allocatedMemForProcessing,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run);

protected:
  const ParallelParams& parallel_params_;

private:
  //initialize data cost for each possible disparity at bottom level
  virtual run_eval::Status initializeDataCosts(const beliefprop::BpSettings& algSettings, const beliefprop::BpLevel& currentBpLevel,
    const std::array<float*, 2>& imagesOnTargetDevice, const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard) = 0;

  //initialize data cost for each possible disparity at levels above the bottom level
  virtual run_eval::Status initializeDataCurrentLevel(const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::BpLevel& prevBpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboardWriteTo,
    unsigned int bp_settings_num_disp_vals) = 0;

  //initialize message values at first level of bp processing to default value
  virtual run_eval::Status initializeMessageValsToDefault(
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    unsigned int bp_settings_num_disp_vals) = 0;

  //run belief propagation processing at current level of processing hierarchy
  virtual run_eval::Status runBPAtCurrentLevel(const beliefprop::BpSettings& algSettings,
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    T* allocatedMemForProcessing) = 0;

  //copy message values from current level of processing to next level of processing
  virtual run_eval::Status copyMessageValuesToNextLevelDown(
    const beliefprop::BpLevel& currentBpLevel,
    const beliefprop::BpLevel& nextBpLevel,
    const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyFrom,
    const beliefprop::CheckerboardMessages<T*>& messagesDeviceCopyTo,
    unsigned int bp_settings_num_disp_vals) = 0;

  //retrieve computed output disparity at each pixel in bottom level using data and message values
  virtual float* retrieveOutputDisparity(
    const beliefprop::BpLevel& BpLevel,
    const beliefprop::DataCostsCheckerboards<T*>& dataCostDeviceCheckerboard,
    const beliefprop::CheckerboardMessages<T*>& messagesDevice,
    unsigned int bp_settings_num_disp_vals) = 0;

  //free memory used for message values in bp processing
  virtual void freeCheckerboardMessagesMemory(const beliefprop::CheckerboardMessages<T*>& checkerboardMessagesToFree,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    std::ranges::for_each(checkerboardMessagesToFree,
      [this, &mem_management_bp_run](auto& checkerboardMessagesSet) {
      mem_management_bp_run->FreeAlignedMemoryOnDevice(checkerboardMessagesSet); });
  }

  //allocate memory for message values in bp processing
  virtual beliefprop::CheckerboardMessages<T*> allocateMemoryForCheckerboardMessages(const unsigned long numDataAllocatePerMessage,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    beliefprop::CheckerboardMessages<T*> outputCheckerboardMessages;
    std::ranges::for_each(outputCheckerboardMessages,
      [this, numDataAllocatePerMessage, &mem_management_bp_run](auto& checkerboardMessagesSet) {
      checkerboardMessagesSet = mem_management_bp_run->AllocateAlignedMemoryOnDevice(numDataAllocatePerMessage, ACCELERATION); });

    return outputCheckerboardMessages;
  }

  //retrieve pointer to bp message data at current level using specified offset
  virtual beliefprop::CheckerboardMessages<T*> retrieveLevelMessageData(
    const beliefprop::CheckerboardMessages<T*>& allCheckerboardMessages, const unsigned long offsetIntoMessages)
  {
    beliefprop::CheckerboardMessages<T*> outputCheckerboardMessages;
    for (unsigned int i = 0; i < outputCheckerboardMessages.size(); i++) {
      outputCheckerboardMessages[i] =
        &((allCheckerboardMessages[i])[offsetIntoMessages]);
    }

    return outputCheckerboardMessages;
  }

  //free memory allocated for data costs in bp processing
  virtual void freeDataCostsMemory(const beliefprop::DataCostsCheckerboards<T*>& dataCostsToFree,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run) {
    mem_management_bp_run->FreeAlignedMemoryOnDevice(dataCostsToFree[0]);
    mem_management_bp_run->FreeAlignedMemoryOnDevice(dataCostsToFree[1]);
  }

  //allocate memory for data costs in bp processing
  virtual beliefprop::DataCostsCheckerboards<T*> allocateMemoryForDataCosts(const unsigned long numDataCostsCheckerboards,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run) {
    return {mem_management_bp_run->AllocateAlignedMemoryOnDevice(numDataCostsCheckerboards, ACCELERATION), 
            mem_management_bp_run->AllocateAlignedMemoryOnDevice(numDataCostsCheckerboards, ACCELERATION)};
  }

  //allocate and organize data cost and message value data at all levels for bp processing
  virtual std::pair<beliefprop::DataCostsCheckerboards<T*>, beliefprop::CheckerboardMessages<T*>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(
    const unsigned long numDataAllocatePerDataCostsMessageDataArray,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    T* dataAllLevels = mem_management_bp_run->AllocateAlignedMemoryOnDevice(10u*numDataAllocatePerDataCostsMessageDataArray, ACCELERATION);
    return organizeDataCostsAndMessageDataAllLevels(dataAllLevels, numDataAllocatePerDataCostsMessageDataArray);
  }

  //organize data cost and message value data at all bp processing levels
  virtual std::pair<beliefprop::DataCostsCheckerboards<T*>, beliefprop::CheckerboardMessages<T*>> organizeDataCostsAndMessageDataAllLevels(
    T* dataAllLevels, const unsigned long numDataAllocatePerDataCostsMessageDataArray)
  {
    beliefprop::DataCostsCheckerboards<T*> dataCostsDeviceCheckerboardAllLevels;
    dataCostsDeviceCheckerboardAllLevels[0] = dataAllLevels;
    dataCostsDeviceCheckerboardAllLevels[1] =
      &(dataCostsDeviceCheckerboardAllLevels[0][1 * (numDataAllocatePerDataCostsMessageDataArray)]);

    beliefprop::CheckerboardMessages<T*> messagesDeviceAllLevels;
    for (unsigned int i = 0; i < messagesDeviceAllLevels.size(); i++) {
      messagesDeviceAllLevels[i] =
        &(dataCostsDeviceCheckerboardAllLevels[0][(i + 2) * (numDataAllocatePerDataCostsMessageDataArray)]);
    }

    return {dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels};
  }

  //free data costs at all levels for bp processing that are all together in a single array
  virtual void freeDataCostsAllDataInSingleArray(const beliefprop::DataCostsCheckerboards<T*>& dataCostsToFree,
    const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
  {
    mem_management_bp_run->FreeAlignedMemoryOnDevice(dataCostsToFree[0]);
  }

  //retrieve pointer to data costs for level using specified offset
  virtual beliefprop::DataCostsCheckerboards<T*> retrieveLevelDataCosts(const beliefprop::DataCostsCheckerboards<T*>& allDataCosts,
    const unsigned long offsetIntoAllDataCosts)
  {
    return {&(allDataCosts[0][offsetIntoAllDataCosts]),
            &(allDataCosts[1][offsetIntoAllDataCosts])};
  }
};

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map on target device
//input is images on target device for computation
//output is disparity map and processing runtimes
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
std::optional<std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>> ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>::operator()(const std::array<float*, 2> & imagesOnTargetDevice,
  const beliefprop::BpSettings& algSettings, const std::array<unsigned int, 2>& width_heightImages, T* allocatedMemForBpProcessingDevice, T* allocatedMemForProcessing,
  const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run)
{
  if (errorCheck() != run_eval::Status::kNoError) { return {}; }

  std::unordered_map<beliefprop::Runtime_Type, std::array<timingType, 2>> startEndTimes;
  std::vector<std::array<timingType, 2>> eachLevelTimingDataCosts(algSettings.num_levels);
  std::vector<std::array<timingType, 2>> eachLevelTimingBP(algSettings.num_levels);
  std::vector<std::array<timingType, 2>> eachLevelTimingCopy(algSettings.num_levels);
  std::chrono::duration<double> totalTimeBpIters{0}, totalTimeCopyData{0}, totalTimeCopyDataKernel{0};

  //start at the "bottom level" and work way up to determine amount of space needed to store data costs
  std::vector<beliefprop::BpLevel> bpBpLevel;
  bpBpLevel.reserve(algSettings.num_levels);

  //set level properties for bottom level that include processing of full image width/height
  bpBpLevel.push_back(beliefprop::BpLevel(width_heightImages, 0, 0, ACCELERATION));

  //compute level properties which includes offset for each data/message array for each level after the bottom level
  for (unsigned int level_num = 1; level_num < algSettings.num_levels; level_num++) {
    //get current level properties from previous level properties
    bpBpLevel.push_back(bpBpLevel[level_num-1].NextBpLevel<T>(algSettings.num_disp_vals));
  }

  startEndTimes[beliefprop::Runtime_Type::kInitSettingsMalloc][0] = std::chrono::system_clock::now();

  //declare and allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
  //as well as the message data used for bp processing
  //each checkerboard holds half of the data and checkerboard 0 includes the pixel in slot (0, 0)
  beliefprop::DataCostsCheckerboards<T*> dataCostsDeviceAllLevels;
  beliefprop::CheckerboardMessages<T*> messagesDeviceAllLevels;

  //data for each array at all levels set to data up to final level (corresponds to offset at final level) plus data amount at final level
  const unsigned long dataAllLevelsEachDataMessageArr = bpBpLevel[algSettings.num_levels-1].LevelProperties().offset_into_arrays_ +
    bpBpLevel[algSettings.num_levels-1].NumDataInBpArrays<T>(algSettings.num_disp_vals);

  //assuming that width includes padding
  if constexpr (bp_params::kUseOptGPUMemManagement) {
    if constexpr (bp_params::kAllocateFreeBpMemoryOutsideRuns) {
      std::tie(dataCostsDeviceAllLevels, messagesDeviceAllLevels) =
        organizeDataCostsAndMessageDataAllLevels(allocatedMemForBpProcessingDevice, dataAllLevelsEachDataMessageArr);
    }
    else {
      //call function that allocates all data in single array and then set offsets in array for data costs and message data locations
      std::tie(dataCostsDeviceAllLevels, messagesDeviceAllLevels) =
        allocateAndOrganizeDataCostsAndMessageDataAllLevels(dataAllLevelsEachDataMessageArr, mem_management_bp_run);
    }
  }
  else {
    dataCostsDeviceAllLevels = allocateMemoryForDataCosts(dataAllLevelsEachDataMessageArr, mem_management_bp_run);
  }

  auto currTime = std::chrono::system_clock::now();
  startEndTimes[beliefprop::Runtime_Type::kInitSettingsMalloc][1] = currTime;
  eachLevelTimingDataCosts[0][0] = currTime;

  //initialize the data cost at the bottom level
  auto errCode = initializeDataCosts(algSettings, bpBpLevel[0], imagesOnTargetDevice, dataCostsDeviceAllLevels);
  if (errCode != run_eval::Status::kNoError) { return {}; }

  currTime = std::chrono::system_clock::now();
  eachLevelTimingDataCosts[0][1] = currTime;
  startEndTimes[beliefprop::Runtime_Type::kDataCostsHigherLevel][0] = currTime;

  //set the data costs at each level from the bottom level "up"
  for (unsigned int level_num = 1u; level_num < algSettings.num_levels; level_num++)
  {
    eachLevelTimingDataCosts[level_num][0] = std::chrono::system_clock::now();
    errCode = initializeDataCurrentLevel(bpBpLevel[level_num], bpBpLevel[level_num - 1],
      retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpBpLevel[level_num - 1u].LevelProperties().offset_into_arrays_),
      retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpBpLevel[level_num].LevelProperties().offset_into_arrays_),
      algSettings.num_disp_vals);
    if (errCode != run_eval::Status::kNoError) { return {}; }

    eachLevelTimingDataCosts[level_num][1] = std::chrono::system_clock::now();
  }

  currTime = eachLevelTimingDataCosts[algSettings.num_levels-1][1];
  startEndTimes[beliefprop::Runtime_Type::kDataCostsHigherLevel][1] = currTime;
  startEndTimes[beliefprop::Runtime_Type::kInitMessages][0] = currTime;

  //get and use offset into data at current processing level of pyramid
  beliefprop::DataCostsCheckerboards<T*> dataCostsDeviceCurrentLevel = retrieveLevelDataCosts(
    dataCostsDeviceAllLevels, bpBpLevel[algSettings.num_levels - 1u].LevelProperties().offset_into_arrays_);

  //declare the space to pass the BP messages; need to have two "sets" of checkerboards because the message
  //data at the "higher" level in the image pyramid need copied to a lower level without overwriting values
  std::array<beliefprop::CheckerboardMessages<T*>, 2> messagesDevice;

  //assuming that width includes padding
  if constexpr (bp_params::kUseOptGPUMemManagement) {
    messagesDevice[0] = retrieveLevelMessageData(messagesDeviceAllLevels, bpBpLevel[algSettings.num_levels - 1u].LevelProperties().offset_into_arrays_);
  }
  else {
    //allocate the space for the message values in the first checkboard set at the current level
    messagesDevice[0] = allocateMemoryForCheckerboardMessages(
      bpBpLevel[algSettings.num_levels - 1u].NumDataInBpArrays<T>(algSettings.num_disp_vals), mem_management_bp_run);
  }

  startEndTimes[beliefprop::Runtime_Type::kInitMessagesKernel][0] = std::chrono::system_clock::now();

  //initialize all the BP message values at every pixel for every disparity to 0
  errCode = initializeMessageValsToDefault(bpBpLevel[algSettings.num_levels - 1u], messagesDevice[0], algSettings.num_disp_vals);
  if (errCode != run_eval::Status::kNoError) { return {}; }

  currTime = std::chrono::system_clock::now();
  startEndTimes[beliefprop::Runtime_Type::kInitMessagesKernel][1] = currTime;
  startEndTimes[beliefprop::Runtime_Type::kInitMessages][1] = currTime;

  //alternate between checkerboard sets 0 and 1
  enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
  Checkerboard_Num currCheckerboardSet{Checkerboard_Num::CHECKERBOARD_ZERO};

  //run BP at each level in the "pyramid" starting on top and continuing to the bottom
  //where the final movement values are computed...the message values are passed from
  //the upper level to the lower levels; this pyramid methods causes the BP message values
  //to converge more quickly
  for (int level_num = (int)algSettings.num_levels - 1; level_num >= 0; level_num--)
  {
    const auto timeBpIterStart = std::chrono::system_clock::now();

    //need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
    errCode = runBPAtCurrentLevel(algSettings, bpBpLevel[(unsigned int)level_num], dataCostsDeviceCurrentLevel, messagesDevice[currCheckerboardSet], allocatedMemForProcessing);
    if (errCode != run_eval::Status::kNoError) { return {}; }

    const auto timeBpIterEnd = std::chrono::system_clock::now();
    totalTimeBpIters += timeBpIterEnd - timeBpIterStart;
    const auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

    //if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
    if (level_num > 0)
    {
      //use offset into allocated memory at next level
      dataCostsDeviceCurrentLevel = retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpBpLevel[level_num - 1].LevelProperties().offset_into_arrays_);

      //assuming that width includes padding
      if constexpr (bp_params::kUseOptGPUMemManagement) {
        messagesDevice[(currCheckerboardSet + 1) % 2] = retrieveLevelMessageData(
          messagesDeviceAllLevels, bpBpLevel[level_num - 1].LevelProperties().offset_into_arrays_);
      }
      else {
        //allocate space in the GPU for the message values in the checkerboard set to copy to
        messagesDevice[(currCheckerboardSet + 1) % 2] = allocateMemoryForCheckerboardMessages(
          bpBpLevel[level_num - 1].NumDataInBpArrays<T>(algSettings.num_disp_vals));
      }

      const auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

      //currentCheckerboardSet = index copying data from
      //(currentCheckerboardSet + 1) % 2 = index copying data to
      errCode = copyMessageValuesToNextLevelDown(bpBpLevel[level_num], bpBpLevel[level_num - 1],
        messagesDevice[currCheckerboardSet], messagesDevice[(currCheckerboardSet + 1) % 2],
        algSettings.num_disp_vals);
      if (errCode != run_eval::Status::kNoError) { return {}; }

      const auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
      totalTimeCopyDataKernel += timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
      eachLevelTimingCopy[level_num][0] = timeCopyMessageValuesKernelStart;
      eachLevelTimingCopy[level_num][1] = timeCopyMessageValuesKernelEnd;

      //assuming that width includes padding
      if constexpr (!bp_params::kUseOptGPUMemManagement) {
        //free the now-copied from computed data of the completed level
        freeCheckerboardMessagesMemory(messagesDevice[currCheckerboardSet], mem_management_bp_run);
      }

      //alternate between checkerboard parts 1 and 2
      currCheckerboardSet = (currCheckerboardSet == Checkerboard_Num::CHECKERBOARD_ZERO) ?
        Checkerboard_Num::CHECKERBOARD_ONE : Checkerboard_Num::CHECKERBOARD_ZERO;
    }

    const auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
    totalTimeCopyData += timeCopyMessageValuesEnd - timeCopyMessageValuesStart;
    eachLevelTimingBP[level_num][0] = timeBpIterStart;
    eachLevelTimingBP[level_num][1] = timeBpIterEnd;
  }

  startEndTimes[beliefprop::Runtime_Type::kOutputDisparity][0] = std::chrono::system_clock::now();

  //assume in bottom level when retrieving output disparity
  float* resultingDisparityMapCompDevice = retrieveOutputDisparity(bpBpLevel[0],
    dataCostsDeviceCurrentLevel, messagesDevice[currCheckerboardSet], algSettings.num_disp_vals);
  if (resultingDisparityMapCompDevice == nullptr) { return {}; }

  currTime = std::chrono::system_clock::now();
  startEndTimes[beliefprop::Runtime_Type::kOutputDisparity][1] = currTime;
  startEndTimes[beliefprop::Runtime_Type::kFinalFree][0] = currTime;

  if constexpr (bp_params::kUseOptGPUMemManagement) {
    if constexpr (bp_params::kAllocateFreeBpMemoryOutsideRuns) {
      //do nothing; memory free outside of runs
    }
    else {
      //now free the allocated data space; all data in single array when
      //bp_params::kUseOptGPUMemManagement set to true
      freeDataCostsAllDataInSingleArray(dataCostsDeviceAllLevels, mem_management_bp_run);
    }
  }
  else {
    //free the device storage allocated to the message values used to retrieve the output movement values
    freeCheckerboardMessagesMemory(messagesDevice[(currCheckerboardSet == 0) ? 0 : 1], mem_management_bp_run);

    //now free the allocated data space
    freeDataCostsMemory(dataCostsDeviceAllLevels, mem_management_bp_run);
  }

  startEndTimes[beliefprop::Runtime_Type::kFinalFree][1] = std::chrono::system_clock::now();

  startEndTimes[beliefprop::Runtime_Type::kLevel0DataCosts] = eachLevelTimingDataCosts[0];
  startEndTimes[beliefprop::Runtime_Type::kLevel0Bp] = eachLevelTimingBP[0];
  startEndTimes[beliefprop::Runtime_Type::kLevel0Copy] = eachLevelTimingCopy[0];
  if (eachLevelTimingBP.size() > 1) {
    startEndTimes[beliefprop::Runtime_Type::kLevel1DataCosts] = eachLevelTimingDataCosts[1];
    startEndTimes[beliefprop::Runtime_Type::kLevel1Bp] = eachLevelTimingBP[1];
    startEndTimes[beliefprop::Runtime_Type::kLevel1Copy] = eachLevelTimingCopy[1];
  }
  if (eachLevelTimingBP.size() > 2) {
    startEndTimes[beliefprop::Runtime_Type::kLevel2DataCosts] = eachLevelTimingDataCosts[2];
    startEndTimes[beliefprop::Runtime_Type::kLevel2Bp] = eachLevelTimingBP[2];
    startEndTimes[beliefprop::Runtime_Type::kLevel2Copy] = eachLevelTimingCopy[2];
  }
  if (eachLevelTimingBP.size() > 3) {
    startEndTimes[beliefprop::Runtime_Type::kLevel3DataCosts] = eachLevelTimingDataCosts[3];
    startEndTimes[beliefprop::Runtime_Type::kLevel3Bp] = eachLevelTimingBP[3];
    startEndTimes[beliefprop::Runtime_Type::kLevel3Copy] = eachLevelTimingCopy[3];
  }
  if (eachLevelTimingBP.size() > 4) {
    startEndTimes[beliefprop::Runtime_Type::kLevel4DataCosts] = eachLevelTimingDataCosts[4];
    startEndTimes[beliefprop::Runtime_Type::kLevel4Bp] = eachLevelTimingBP[4];
    startEndTimes[beliefprop::Runtime_Type::kLevel4Copy] = eachLevelTimingCopy[4];
  }
  if (eachLevelTimingBP.size() > 5) {
    startEndTimes[beliefprop::Runtime_Type::kLevel5DataCosts] = eachLevelTimingDataCosts[5];
    startEndTimes[beliefprop::Runtime_Type::kLevel5Bp] = eachLevelTimingBP[5];
    startEndTimes[beliefprop::Runtime_Type::kLevel5Copy] = eachLevelTimingCopy[5];
  }
  if (eachLevelTimingBP.size() > 6) {
    startEndTimes[beliefprop::Runtime_Type::kLevel6DataCosts] = eachLevelTimingDataCosts[6];
    startEndTimes[beliefprop::Runtime_Type::kLevel6Bp] = eachLevelTimingBP[6];
    startEndTimes[beliefprop::Runtime_Type::kLevel6Copy] = eachLevelTimingCopy[6];
  }
  if (eachLevelTimingBP.size() > 7) {
    startEndTimes[beliefprop::Runtime_Type::kLevel7DataCosts] = eachLevelTimingDataCosts[7];
    startEndTimes[beliefprop::Runtime_Type::kLevel7Bp] = eachLevelTimingBP[7];
    startEndTimes[beliefprop::Runtime_Type::kLevel7Copy] = eachLevelTimingCopy[7];
  }
  if (eachLevelTimingBP.size() > 8) {
    startEndTimes[beliefprop::Runtime_Type::kLevel8DataCosts] = eachLevelTimingDataCosts[8];
    startEndTimes[beliefprop::Runtime_Type::kLevel8Bp] = eachLevelTimingBP[8];
    startEndTimes[beliefprop::Runtime_Type::kLevel8Copy] = eachLevelTimingCopy[8];
  }
  if (eachLevelTimingBP.size() > 9) {
    startEndTimes[beliefprop::Runtime_Type::kLevel9DataCosts] = eachLevelTimingDataCosts[9];
    startEndTimes[beliefprop::Runtime_Type::kLevel9Bp] = eachLevelTimingBP[9];
    startEndTimes[beliefprop::Runtime_Type::kLevel9Copy] = eachLevelTimingCopy[9];
  }

  //add timing for each runtime segment to segmentTimings object
  DetailedTimings segmentTimings(beliefprop::kTimingNames);
  std::ranges::for_each(startEndTimes,
    [&segmentTimings](const auto& currentRuntimeNameAndTiming) {
    segmentTimings.AddTiming(currentRuntimeNameAndTiming.first,
      currentRuntimeNameAndTiming.second[1] - currentRuntimeNameAndTiming.second[0]);
  });

  segmentTimings.AddTiming(beliefprop::Runtime_Type::kBpIters, totalTimeBpIters);
  segmentTimings.AddTiming(beliefprop::Runtime_Type::kCopyData, totalTimeCopyData);
  segmentTimings.AddTiming(beliefprop::Runtime_Type::kCopyDataKernel, totalTimeCopyDataKernel);

  const auto totalTimed = segmentTimings.MedianTiming(beliefprop::Runtime_Type::kInitSettingsMalloc)
    + segmentTimings.MedianTiming(beliefprop::Runtime_Type::kLevel0DataCosts)
    + segmentTimings.MedianTiming(beliefprop::Runtime_Type::kDataCostsHigherLevel) + segmentTimings.MedianTiming(beliefprop::Runtime_Type::kInitMessages)
    + totalTimeBpIters + totalTimeCopyData + segmentTimings.MedianTiming(beliefprop::Runtime_Type::kOutputDisparity)
    + segmentTimings.MedianTiming(beliefprop::Runtime_Type::kFinalFree);
  segmentTimings.AddTiming(beliefprop::Runtime_Type::kTotalTimed, totalTimed);

  return std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>{resultingDisparityMapCompDevice, segmentTimings};
}

#endif /* PROCESSBPONTARGETDEVICE_H_ */
