/*
 * ProcessBPOnTargetDevice.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICE_H_
#define PROCESSBPONTARGETDEVICE_H_

#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include <math.h>
#include <chrono>
#include <unordered_map>
#include "../RuntimeTiming/DetailedTimings.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"
#include <memory>
#include <tuple>
#include <array>

typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

template<typename T, typename U, typename V=float*>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }
	virtual ~ProcessBPOnTargetDevice() { }

		virtual int getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize)
		{
			return (int)ceil(((float)widthLevelActualIntegerSize) / 2.0);
		}

		virtual void allocateRawMemoryOnTargetDevice(void** arrayToAllocate, unsigned long numBytesAllocate) = 0;

		virtual U allocateMemoryOnTargetDevice(unsigned long numData) = 0;

		virtual void freeRawMemoryOnTargetDevice(void* arrayToFree) = 0;

		virtual void freeMemoryOnTargetDevice(U memoryToFree) = 0;

		virtual void initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
				const std::array<V, 2>& imagesOnTargetDevice, const dataCostData<U>& dataCostDeviceCheckerboard) = 0;

		virtual void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
				const levelProperties& prevLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const dataCostData<U>& dataCostDeviceCheckerboardWriteTo) = 0;

		virtual void initializeMessageValsToDefault(
				const levelProperties& currentLevelProperties,
				const checkerboardMessages<U>& messagesDevice) = 0;

		virtual void runBPAtCurrentLevel(const BPsettings& algSettings,
				const levelProperties& currentLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const checkerboardMessages<U>& messagesDevice) = 0;

		virtual void copyMessageValuesToNextLevelDown(
				const levelProperties& currentLevelProperties,
				const levelProperties& nextlevelProperties,
				const checkerboardMessages<U>& messagesDeviceCopyFrom,
				const checkerboardMessages<U>& messagesDeviceCopyTo) = 0;

		virtual V retrieveOutputDisparity(
				const levelProperties& levelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const checkerboardMessages<U>& messagesDevice) = 0;

		virtual int getPaddedCheckerboardWidth(int checkerboardWidth);

		unsigned long getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements);

		virtual void freeCheckerboardMessagesMemory(const checkerboardMessages<U>& checkerboardMessagesToFree)
		{
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesU_Checkerboard0);
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesD_Checkerboard0);
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesL_Checkerboard0);
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesR_Checkerboard0);

			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesU_Checkerboard1);
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesD_Checkerboard1);
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesL_Checkerboard1);
			freeMemoryOnTargetDevice(checkerboardMessagesToFree.messagesR_Checkerboard1);
		}

		virtual checkerboardMessages<U> allocateMemoryForCheckerboardMessages(unsigned long numDataAllocatePerMessage)
		{
			checkerboardMessages<U> outputCheckerboardMessages;

			outputCheckerboardMessages.messagesU_Checkerboard0 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);
			outputCheckerboardMessages.messagesD_Checkerboard0 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);
			outputCheckerboardMessages.messagesL_Checkerboard0 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);
			outputCheckerboardMessages.messagesR_Checkerboard0 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);

			outputCheckerboardMessages.messagesU_Checkerboard1 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);
			outputCheckerboardMessages.messagesD_Checkerboard1 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);
			outputCheckerboardMessages.messagesL_Checkerboard1 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);
			outputCheckerboardMessages.messagesR_Checkerboard1 = allocateMemoryOnTargetDevice(numDataAllocatePerMessage);

			return outputCheckerboardMessages;
		}

		virtual checkerboardMessages<U> retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(const checkerboardMessages<U>& allCheckerboardMessages, unsigned long offsetIntoAllCheckerboardMessages)
		{
			checkerboardMessages<U> outputCheckerboardMessages;

			outputCheckerboardMessages.messagesU_Checkerboard0 = &(allCheckerboardMessages.messagesU_Checkerboard0[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesD_Checkerboard0 = &(allCheckerboardMessages.messagesD_Checkerboard0[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesL_Checkerboard0 = &(allCheckerboardMessages.messagesL_Checkerboard0[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesR_Checkerboard0 = &(allCheckerboardMessages.messagesR_Checkerboard0[offsetIntoAllCheckerboardMessages]);

			outputCheckerboardMessages.messagesU_Checkerboard1 = &(allCheckerboardMessages.messagesU_Checkerboard1[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesD_Checkerboard1 = &(allCheckerboardMessages.messagesD_Checkerboard1[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesL_Checkerboard1 = &(allCheckerboardMessages.messagesL_Checkerboard1[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesR_Checkerboard1 = &(allCheckerboardMessages.messagesR_Checkerboard1[offsetIntoAllCheckerboardMessages]);

			return outputCheckerboardMessages;
		}

		virtual void freeDataCostsMemory(const dataCostData<U>& dataCostsToFree)
		{
			freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0);
			freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard1);
		}

		virtual dataCostData<U> allocateMemoryForDataCosts(unsigned long numDataAllocatePerDataCostsCheckerboard)
		{
			dataCostData<U> outputDataCosts;

			outputDataCosts.dataCostCheckerboard0 = allocateMemoryOnTargetDevice(numDataAllocatePerDataCostsCheckerboard);
			outputDataCosts.dataCostCheckerboard1 = allocateMemoryOnTargetDevice(numDataAllocatePerDataCostsCheckerboard);

			return outputDataCosts;
		}

		virtual std::pair<dataCostData<U>, checkerboardMessages<U>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(unsigned long numDataAllocatePerDataCostsMessageDataArray)
		{
			dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
			checkerboardMessages<U> messagesDeviceAllLevels;

			dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0 = allocateMemoryOnTargetDevice(10*numDataAllocatePerDataCostsMessageDataArray);
			dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard1 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[1 * (numDataAllocatePerDataCostsMessageDataArray)]);

			messagesDeviceAllLevels.messagesU_Checkerboard0 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[2 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceAllLevels.messagesD_Checkerboard0 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[3 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceAllLevels.messagesL_Checkerboard0 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[4 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceAllLevels.messagesR_Checkerboard0 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[5 * (numDataAllocatePerDataCostsMessageDataArray)]);

			messagesDeviceAllLevels.messagesU_Checkerboard1 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[6 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceAllLevels.messagesD_Checkerboard1 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[7 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceAllLevels.messagesL_Checkerboard1 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[8 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceAllLevels.messagesR_Checkerboard1 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[9 * (numDataAllocatePerDataCostsMessageDataArray)]);

			return std::make_pair(dataCostsDeviceCheckerboardAllLevels, messagesDeviceAllLevels);
		}

		virtual void freeDataCostsAllDataInSingleArray(const dataCostData<U>& dataCostsToFree)
		{
			freeMemoryOnTargetDevice(dataCostsToFree.dataCostCheckerboard0);
		}

		virtual dataCostData<U> retrieveCurrentDataCostsFromOffsetIntoAllDataCosts(const dataCostData<U>& allDataCosts, unsigned long offsetIntoAllDataCosts)
		{
			dataCostData<U> outputDataCosts;

			outputDataCosts.dataCostCheckerboard0 = &(allDataCosts.dataCostCheckerboard0[offsetIntoAllDataCosts]);
			outputDataCosts.dataCostCheckerboard1 = &(allDataCosts.dataCostCheckerboard1[offsetIntoAllDataCosts]);

			return outputDataCosts;
		}

		//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
		//input is images image1Pixels and image1Pixels
		//output is resultingDisparityMap
		std::pair<V, DetailedTimings<Runtime_Type_BP>> operator()(const std::array<V, 2>& imagesOnTargetDevice,
				const BPsettings& algSettings, unsigned int widthImages, unsigned int heightImages);
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
