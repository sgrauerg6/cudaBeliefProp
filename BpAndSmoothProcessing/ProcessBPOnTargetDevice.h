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
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../RuntimeTiming/DetailedTimings.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"
#include "RunBpStereoSetMemoryManagement.h"

typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

//Abstract class to process belief propagation on target device
//Some of the class functions need to be overridden to for processing on
//target device
template<typename T, typename U, unsigned int DISP_VALS, beliefprop::AccSetting ACC_SETTING, typename V=float*>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }

	virtual beliefprop::Status initializeDataCosts(const beliefprop::BPsettings& algSettings, const beliefprop::levelProperties& currentLevelProperties,
			const std::array<V, 2>& imagesOnTargetDevice, const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard) = 0;

	virtual beliefprop::Status initializeDataCurrentLevel(const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::levelProperties& prevLevelProperties,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboardWriteTo,
			const unsigned int bpSettingsNumDispVals) = 0;

	virtual beliefprop::Status initializeMessageValsToDefault(
			const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::checkerboardMessages<U>& messagesDevice,
			const unsigned int bpSettingsNumDispVals) = 0;

	virtual beliefprop::Status runBPAtCurrentLevel(const beliefprop::BPsettings& algSettings,
			const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
			const beliefprop::checkerboardMessages<U>& messagesDevice,
			U allocatedMemForProcessing) = 0;

	virtual beliefprop::Status copyMessageValuesToNextLevelDown(
			const beliefprop::levelProperties& currentLevelProperties,
			const beliefprop::levelProperties& nextlevelProperties,
			const beliefprop::checkerboardMessages<U>& messagesDeviceCopyFrom,
			const beliefprop::checkerboardMessages<U>& messagesDeviceCopyTo,
			const unsigned int bpSettingsNumDispVals) = 0;

	virtual V retrieveOutputDisparity(
			const beliefprop::levelProperties& levelProperties,
			const beliefprop::dataCostData<U>& dataCostDeviceCheckerboard,
			const beliefprop::checkerboardMessages<U>& messagesDevice,
			const unsigned int bpSettingsNumDispVals) = 0;
	
	virtual beliefprop::Status errorCheck(const char *file = "", int line = 0, bool abort = false) const {
		return beliefprop::Status::NO_ERROR;
	}

	virtual void freeCheckerboardMessagesMemory(const beliefprop::checkerboardMessages<U>& checkerboardMessagesToFree,
	const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun)
	{
		std::for_each(checkerboardMessagesToFree.checkerboardMessagesAtLevel_.begin(), checkerboardMessagesToFree.checkerboardMessagesAtLevel_.end(),
			[this, &memManagementBpRun](auto& checkerboardMessagesSet) {
			memManagementBpRun->freeAlignedMemoryOnDevice(checkerboardMessagesSet); });
	}

	virtual beliefprop::checkerboardMessages<U> allocateMemoryForCheckerboardMessages(const unsigned long numDataAllocatePerMessage,
		const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun)
	{
		beliefprop::checkerboardMessages<U> outputCheckerboardMessages;
		std::for_each(outputCheckerboardMessages.checkerboardMessagesAtLevel_.begin(), outputCheckerboardMessages.checkerboardMessagesAtLevel_.end(),
			[this, numDataAllocatePerMessage, &memManagementBpRun](auto& checkerboardMessagesSet) {
			checkerboardMessagesSet = memManagementBpRun->allocateAlignedMemoryOnDevice(numDataAllocatePerMessage, ACC_SETTING); });

		return outputCheckerboardMessages;
	}

	virtual beliefprop::checkerboardMessages<U> retrieveLevelMessageData(const beliefprop::checkerboardMessages<U>& allCheckerboardMessages, const unsigned long offsetIntoMessages)
	{
		beliefprop::checkerboardMessages<U> outputCheckerboardMessages;
		for (unsigned int i = 0; i < outputCheckerboardMessages.checkerboardMessagesAtLevel_.size(); i++) {
			outputCheckerboardMessages.checkerboardMessagesAtLevel_[i] =
				&((allCheckerboardMessages.checkerboardMessagesAtLevel_[i])[offsetIntoMessages]);
		}

		return outputCheckerboardMessages;
	}

	virtual void freeDataCostsMemory(const beliefprop::dataCostData<U>& dataCostsToFree,
		const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun) {
		memManagementBpRun->freeAlignedMemoryOnDevice(dataCostsToFree.dataCostCheckerboard0_);
		memManagementBpRun->freeAlignedMemoryOnDevice(dataCostsToFree.dataCostCheckerboard1_);
	}

	virtual beliefprop::dataCostData<U> allocateMemoryForDataCosts(const unsigned long numDataCostsCheckerboard,
	    const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun) {
		return {memManagementBpRun->allocateAlignedMemoryOnDevice(numDataCostsCheckerboard, ACC_SETTING), 
				memManagementBpRun->allocateAlignedMemoryOnDevice(numDataCostsCheckerboard, ACC_SETTING)};
	}

	virtual std::pair<beliefprop::dataCostData<U>, beliefprop::checkerboardMessages<U>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(
			const unsigned long numDataAllocatePerDataCostsMessageDataArray,
			const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun)
	{
		U dataAllLevels = memManagementBpRun->allocateAlignedMemoryOnDevice(10u*numDataAllocatePerDataCostsMessageDataArray, ACC_SETTING);
		return organizeDataCostsAndMessageDataAllLevels(dataAllLevels, numDataAllocatePerDataCostsMessageDataArray);
	}

	virtual std::pair<beliefprop::dataCostData<U>, beliefprop::checkerboardMessages<U>> organizeDataCostsAndMessageDataAllLevels(
			U dataAllLevels, const unsigned long numDataAllocatePerDataCostsMessageDataArray)
	{
		beliefprop::dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_ = dataAllLevels;
		dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard1_ =
				&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_[1 * (numDataAllocatePerDataCostsMessageDataArray)]);

		beliefprop::checkerboardMessages<U> messagesDeviceAllLevels;
		for (unsigned int i = 0; i < messagesDeviceAllLevels.checkerboardMessagesAtLevel_.size(); i++) {
			messagesDeviceAllLevels.checkerboardMessagesAtLevel_[i] =
					&(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0_[(i + 2) * (numDataAllocatePerDataCostsMessageDataArray)]);
		}

		return {dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels};
	}

	virtual void freeDataCostsAllDataInSingleArray(const beliefprop::dataCostData<U>& dataCostsToFree,
	const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun) {
		memManagementBpRun->freeAlignedMemoryOnDevice(dataCostsToFree.dataCostCheckerboard0_);
	}

	virtual beliefprop::dataCostData<U> retrieveLevelDataCosts(const beliefprop::dataCostData<U>& allDataCosts, const unsigned long offsetIntoAllDataCosts) {
		return {&(allDataCosts.dataCostCheckerboard0_[offsetIntoAllDataCosts]),
			    &(allDataCosts.dataCostCheckerboard1_[offsetIntoAllDataCosts])};
	}

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	std::pair<V, DetailedTimings<Runtime_Type_BP>> operator()(const std::array<V, 2>& imagesOnTargetDevice,
			const beliefprop::BPsettings& algSettings, const std::array<unsigned int, 2>& widthHeightImages,
			U allocatedMemForBpProcessingDevice, U allocatedMemForProcessing,
			const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun);
};

//run the belief propagation algorithm with on a set of stereo images to generate a disparity map on target device
//input is images on target device for computation
//output is disparity map and processing runtimes
template<typename T, typename U, unsigned int DISP_VALS, beliefprop::AccSetting ACC_SETTING, typename V>
std::pair<V, DetailedTimings<Runtime_Type_BP>> ProcessBPOnTargetDevice<T, U, DISP_VALS, ACC_SETTING, V>::operator()(const std::array<V, 2> & imagesOnTargetDevice,
	const beliefprop::BPsettings& algSettings, const std::array<unsigned int, 2>& widthHeightImages, U allocatedMemForBpProcessingDevice, U allocatedMemForProcessing,
	const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun)
{
	if (errorCheck() != beliefprop::Status::NO_ERROR) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> startEndTimes;
	std::vector<std::pair<timingType, timingType>> eachLevelTimingDataCosts(algSettings.numLevels_);
	std::vector<std::pair<timingType, timingType>> eachLevelTimingBP(algSettings.numLevels_);
	std::vector<std::pair<timingType, timingType>> eachLevelTimingCopy(algSettings.numLevels_);
	double totalTimeBpIters{0.0}, totalTimeCopyData{0.0}, totalTimeCopyDataKernel{0.0};

	//start at the "bottom level" and work way up to determine amount of space needed to store data costs
	std::vector<beliefprop::levelProperties> bpLevelProperties;
	bpLevelProperties.reserve(algSettings.numLevels_);

	//set level properties for bottom level that include processing of full image width/height
	bpLevelProperties.push_back(beliefprop::levelProperties(widthHeightImages, 0, 0, ACC_SETTING));

	//compute level properties which includes offset for each data/message array for each level after the bottom level
	for (unsigned int levelNum = 1; levelNum < algSettings.numLevels_; levelNum++) {
		//get current level properties from previous level properties
		bpLevelProperties.push_back(bpLevelProperties[levelNum-1].getNextLevelProperties<T>(algSettings.numDispVals_));
	}

	startEndTimes[Runtime_Type_BP::INIT_SETTINGS_MALLOC].first = std::chrono::system_clock::now();

	//declare and allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//as well as the message data used for bp processing
	//each checkerboard holds half of the data and checkerboard 0 includes the pixel in slot (0, 0)
	beliefprop::dataCostData<U> dataCostsDeviceAllLevels;
	beliefprop::checkerboardMessages<U> messagesDeviceAllLevels;

	//data for each array at all levels set to data up to final level (corresponds to offset at final level) plus data amount at final level
	const unsigned long dataAllLevelsEachDataMessageArr = bpLevelProperties[algSettings.numLevels_-1].offsetIntoArrays_ +
			bpLevelProperties[algSettings.numLevels_-1].getNumDataInBpArrays<T>(algSettings.numDispVals_);

	//assuming that width includes padding
	if constexpr (beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT) {
		if constexpr (beliefprop::ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
			std::tie(dataCostsDeviceAllLevels, messagesDeviceAllLevels) =
					organizeDataCostsAndMessageDataAllLevels(allocatedMemForBpProcessingDevice, dataAllLevelsEachDataMessageArr);
		}
		else {
			//call function that allocates all data in single array and then set offsets in array for data costs and message data locations
			std::tie(dataCostsDeviceAllLevels, messagesDeviceAllLevels) =
					allocateAndOrganizeDataCostsAndMessageDataAllLevels(dataAllLevelsEachDataMessageArr, memManagementBpRun);
		}
	}
	else {
		dataCostsDeviceAllLevels = allocateMemoryForDataCosts(dataAllLevelsEachDataMessageArr, memManagementBpRun);
	}

    auto currTime = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::INIT_SETTINGS_MALLOC].second = currTime;
	eachLevelTimingDataCosts[0].first = currTime;

	//initialize the data cost at the bottom level
	auto errCode = initializeDataCosts(algSettings, bpLevelProperties[0], imagesOnTargetDevice, dataCostsDeviceAllLevels);
	if (errCode != beliefprop::Status::NO_ERROR) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

    currTime = std::chrono::system_clock::now();
	eachLevelTimingDataCosts[0].second = currTime;
	startEndTimes[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].first = currTime;

	//set the data costs at each level from the bottom level "up"
	for (unsigned int levelNum = 1u; levelNum < algSettings.numLevels_; levelNum++)
	{
		eachLevelTimingDataCosts[levelNum].first = std::chrono::system_clock::now();
		errCode = initializeDataCurrentLevel(bpLevelProperties[levelNum], bpLevelProperties[levelNum - 1],
				    retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpLevelProperties[levelNum - 1u].offsetIntoArrays_),
				    retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpLevelProperties[levelNum].offsetIntoArrays_),
				    algSettings.numDispVals_);
		if (errCode != beliefprop::Status::NO_ERROR) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

		eachLevelTimingDataCosts[levelNum].second = std::chrono::system_clock::now();
	}

    currTime = eachLevelTimingDataCosts[algSettings.numLevels_-1].second;
	startEndTimes[Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL].second = currTime;
	startEndTimes[Runtime_Type_BP::INIT_MESSAGES].first = currTime;

	//get and use offset into data at current processing level of pyramid
	beliefprop::dataCostData<U> dataCostsDeviceCurrentLevel = retrieveLevelDataCosts(
			dataCostsDeviceAllLevels, bpLevelProperties[algSettings.numLevels_ - 1u].offsetIntoArrays_);

	//declare the space to pass the BP messages; need to have two "sets" of checkerboards because the message
	//data at the "higher" level in the image pyramid need copied to a lower level without overwriting values
	std::array<beliefprop::checkerboardMessages<U>, 2> messagesDevice;

	//assuming that width includes padding
	if constexpr (beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT) {
		messagesDevice[0] = retrieveLevelMessageData(messagesDeviceAllLevels, bpLevelProperties[algSettings.numLevels_ - 1u].offsetIntoArrays_);
	}
	else {
		//allocate the space for the message values in the first checkboard set at the current level
		messagesDevice[0] = allocateMemoryForCheckerboardMessages(
				bpLevelProperties[algSettings.numLevels_ - 1u].getNumDataInBpArrays<T>(algSettings.numDispVals_), memManagementBpRun);
	}

	startEndTimes[Runtime_Type_BP::INIT_MESSAGES_KERNEL].first = std::chrono::system_clock::now();

	//initialize all the BP message values at every pixel for every disparity to 0
	errCode = initializeMessageValsToDefault(bpLevelProperties[algSettings.numLevels_ - 1u], messagesDevice[0], algSettings.numDispVals_);
	if (errCode != beliefprop::Status::NO_ERROR) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

    currTime = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::INIT_MESSAGES_KERNEL].second = currTime;
	startEndTimes[Runtime_Type_BP::INIT_MESSAGES].second = currTime;

	//alternate between checkerboard sets 0 and 1
	enum Checkerboard_Num { CHECKERBOARD_ZERO = 0, CHECKERBOARD_ONE = 1 };
	Checkerboard_Num currCheckerboardSet{Checkerboard_Num::CHECKERBOARD_ZERO};

	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = (int)algSettings.numLevels_ - 1; levelNum >= 0; levelNum--)
	{
		std::chrono::duration<double> diff;
		const auto timeBpIterStart = std::chrono::system_clock::now();

		//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
		errCode = runBPAtCurrentLevel(algSettings, bpLevelProperties[(unsigned int)levelNum], dataCostsDeviceCurrentLevel, messagesDevice[currCheckerboardSet], allocatedMemForProcessing);
   	    if (errCode != beliefprop::Status::NO_ERROR) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

		const auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd - timeBpIterStart;
		totalTimeBpIters += diff.count();
		const auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
		if (levelNum > 0)
		{
			//use offset into allocated memory at next level
			dataCostsDeviceCurrentLevel = retrieveLevelDataCosts(dataCostsDeviceAllLevels, bpLevelProperties[levelNum - 1].offsetIntoArrays_);

			//assuming that width includes padding
			if constexpr (beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT) {
				messagesDevice[(currCheckerboardSet + 1) % 2] = retrieveLevelMessageData(
						messagesDeviceAllLevels, bpLevelProperties[levelNum - 1].offsetIntoArrays_);
			}
			else {
				//allocate space in the GPU for the message values in the checkerboard set to copy to
				messagesDevice[(currCheckerboardSet + 1) % 2] = allocateMemoryForCheckerboardMessages(
						bpLevelProperties[levelNum - 1].getNumDataInBpArrays<T>(algSettings.numDispVals_));
			}

			const auto timeCopyMessageValuesKernelStart = std::chrono::system_clock::now();

			//currentCheckerboardSet = index copying data from
			//(currentCheckerboardSet + 1) % 2 = index copying data to
			errCode = copyMessageValuesToNextLevelDown(bpLevelProperties[levelNum], bpLevelProperties[levelNum - 1],
					messagesDevice[currCheckerboardSet], messagesDevice[(currCheckerboardSet + 1) % 2],
					algSettings.numDispVals_);
     	    if (errCode != beliefprop::Status::NO_ERROR) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

			const auto timeCopyMessageValuesKernelEnd = std::chrono::system_clock::now();
			diff = timeCopyMessageValuesKernelEnd - timeCopyMessageValuesKernelStart;
			totalTimeCopyDataKernel += diff.count();
    		eachLevelTimingCopy[levelNum].first = timeCopyMessageValuesKernelStart;
 	    	eachLevelTimingCopy[levelNum].second = timeCopyMessageValuesKernelEnd;

			//assuming that width includes padding
			if constexpr (!beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT) {
				//free the now-copied from computed data of the completed level
				freeCheckerboardMessagesMemory(messagesDevice[currCheckerboardSet], memManagementBpRun);
			}

			//alternate between checkerboard parts 1 and 2
			currCheckerboardSet = (currCheckerboardSet == Checkerboard_Num::CHECKERBOARD_ZERO) ?
					Checkerboard_Num::CHECKERBOARD_ONE : Checkerboard_Num::CHECKERBOARD_ZERO;
		}

		const auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd - timeCopyMessageValuesStart;
		totalTimeCopyData += diff.count();
		eachLevelTimingBP[levelNum].first = timeBpIterStart;
		eachLevelTimingBP[levelNum].second = timeBpIterEnd;
	}

	startEndTimes[Runtime_Type_BP::OUTPUT_DISPARITY].first = std::chrono::system_clock::now();

	//assume in bottom level when retrieving output disparity
	const V resultingDisparityMapCompDevice = retrieveOutputDisparity(bpLevelProperties[0],
			dataCostsDeviceCurrentLevel, messagesDevice[currCheckerboardSet], algSettings.numDispVals_);
    if (resultingDisparityMapCompDevice == nullptr) { return {nullptr, DetailedTimings<Runtime_Type_BP>()}; }

    currTime = std::chrono::system_clock::now();
	startEndTimes[Runtime_Type_BP::OUTPUT_DISPARITY].second = currTime;
	startEndTimes[Runtime_Type_BP::FINAL_FREE].first = currTime;

	if constexpr (beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT) {
		if constexpr (beliefprop::ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
        	//do nothing; memory free outside of runs
		}
		else {
			//now free the allocated data space; all data in single array when
			//beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT set to true
			freeDataCostsAllDataInSingleArray(dataCostsDeviceAllLevels, memManagementBpRun);
		}
	}
	else {
		//free the device storage allocated to the message values used to retrieve the output movement values
		freeCheckerboardMessagesMemory(messagesDevice[(currCheckerboardSet == 0) ? 0 : 1], memManagementBpRun);

		//now free the allocated data space
		freeDataCostsMemory(dataCostsDeviceAllLevels, memManagementBpRun);
	}

	startEndTimes[Runtime_Type_BP::FINAL_FREE].second = std::chrono::system_clock::now();

	startEndTimes[Runtime_Type_BP::LEVEL_0_DATA_COSTS] = eachLevelTimingDataCosts[0];
    startEndTimes[Runtime_Type_BP::LEVEL_0_BP] = eachLevelTimingBP[0];
    startEndTimes[Runtime_Type_BP::LEVEL_0_COPY] = eachLevelTimingCopy[0];
	if (eachLevelTimingBP.size() > 1) {
	  startEndTimes[Runtime_Type_BP::LEVEL_1_DATA_COSTS] = eachLevelTimingDataCosts[1];
	  startEndTimes[Runtime_Type_BP::LEVEL_1_BP] = eachLevelTimingBP[1];
	  startEndTimes[Runtime_Type_BP::LEVEL_1_COPY] = eachLevelTimingCopy[1];
	}
	if (eachLevelTimingBP.size() > 2) {
	  startEndTimes[Runtime_Type_BP::LEVEL_2_DATA_COSTS] = eachLevelTimingDataCosts[2];
	  startEndTimes[Runtime_Type_BP::LEVEL_2_BP] = eachLevelTimingBP[2];
	  startEndTimes[Runtime_Type_BP::LEVEL_2_COPY] = eachLevelTimingCopy[2];
	}
	if (eachLevelTimingBP.size() > 3) {
	  startEndTimes[Runtime_Type_BP::LEVEL_3_DATA_COSTS] = eachLevelTimingDataCosts[3];
	  startEndTimes[Runtime_Type_BP::LEVEL_3_BP] = eachLevelTimingBP[3];
	  startEndTimes[Runtime_Type_BP::LEVEL_3_COPY] = eachLevelTimingCopy[3];
	}
	if (eachLevelTimingBP.size() > 4) {
	  startEndTimes[Runtime_Type_BP::LEVEL_4_DATA_COSTS] = eachLevelTimingDataCosts[4];
	  startEndTimes[Runtime_Type_BP::LEVEL_4_BP] = eachLevelTimingBP[4];
	  startEndTimes[Runtime_Type_BP::LEVEL_4_COPY] = eachLevelTimingCopy[4];
	}
	if (eachLevelTimingBP.size() > 5) {
	  startEndTimes[Runtime_Type_BP::LEVEL_5_DATA_COSTS] = eachLevelTimingDataCosts[5];
	  startEndTimes[Runtime_Type_BP::LEVEL_5_BP] = eachLevelTimingBP[5];
	  startEndTimes[Runtime_Type_BP::LEVEL_5_COPY] = eachLevelTimingCopy[5];
	}
	if (eachLevelTimingBP.size() > 6) {
	  startEndTimes[Runtime_Type_BP::LEVEL_6_DATA_COSTS] = eachLevelTimingDataCosts[6];
	  startEndTimes[Runtime_Type_BP::LEVEL_6_BP] = eachLevelTimingBP[6];
	  startEndTimes[Runtime_Type_BP::LEVEL_6_COPY] = eachLevelTimingCopy[6];
	}
	if (eachLevelTimingBP.size() > 7) {
	  startEndTimes[Runtime_Type_BP::LEVEL_7_DATA_COSTS] = eachLevelTimingDataCosts[7];
	  startEndTimes[Runtime_Type_BP::LEVEL_7_BP] = eachLevelTimingBP[7];
	  startEndTimes[Runtime_Type_BP::LEVEL_7_COPY] = eachLevelTimingCopy[7];
	}
	if (eachLevelTimingBP.size() > 8) {
	  startEndTimes[Runtime_Type_BP::LEVEL_8_DATA_COSTS] = eachLevelTimingDataCosts[8];
	  startEndTimes[Runtime_Type_BP::LEVEL_8_BP] = eachLevelTimingBP[8];
	  startEndTimes[Runtime_Type_BP::LEVEL_8_COPY] = eachLevelTimingCopy[8];
	}
	if (eachLevelTimingBP.size() > 9) {
	  startEndTimes[Runtime_Type_BP::LEVEL_9_DATA_COSTS] = eachLevelTimingDataCosts[9];
	  startEndTimes[Runtime_Type_BP::LEVEL_9_BP] = eachLevelTimingBP[9];
	  startEndTimes[Runtime_Type_BP::LEVEL_9_COPY] = eachLevelTimingCopy[9];
	}

	//add timing for each runtime segment to segmentTimings object
	DetailedTimings<Runtime_Type_BP> segmentTimings(timingNames_BP);
	std::for_each(startEndTimes.begin(), startEndTimes.end(),
		[&segmentTimings](const auto& currentRuntimeNameAndTiming) {
		segmentTimings.addTiming(currentRuntimeNameAndTiming.first,
			(timingInSecondsDoublePrecision(currentRuntimeNameAndTiming.second.second - currentRuntimeNameAndTiming.second.first)).count());
	});

	segmentTimings.addTiming(Runtime_Type_BP::BP_ITERS, totalTimeBpIters);
	segmentTimings.addTiming(Runtime_Type_BP::COPY_DATA, totalTimeCopyData);
	segmentTimings.addTiming(Runtime_Type_BP::COPY_DATA_KERNEL, totalTimeCopyDataKernel);

	const double totalTimed = segmentTimings.getMedianTiming(Runtime_Type_BP::INIT_SETTINGS_MALLOC)
		+ segmentTimings.getMedianTiming(Runtime_Type_BP::LEVEL_0_DATA_COSTS)
		+ segmentTimings.getMedianTiming(Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL) + segmentTimings.getMedianTiming(Runtime_Type_BP::INIT_MESSAGES)
		+ totalTimeBpIters + totalTimeCopyData + segmentTimings.getMedianTiming(Runtime_Type_BP::OUTPUT_DISPARITY)
		+ segmentTimings.getMedianTiming(Runtime_Type_BP::FINAL_FREE);
	segmentTimings.addTiming(Runtime_Type_BP::TOTAL_TIMED, totalTimed);

	return {resultingDisparityMapCompDevice, segmentTimings};
}


#endif /* PROCESSBPONTARGETDEVICE_H_ */
