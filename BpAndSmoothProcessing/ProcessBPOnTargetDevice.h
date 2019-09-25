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

template<typename T, typename U, typename V=float*>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }
	virtual ~ProcessBPOnTargetDevice() { }

		virtual int getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize)
		{
			return (int)ceil(((float)widthLevelActualIntegerSize) / 2.0);
		}

		virtual void allocateMemoryOnTargetDevice(void** arrayToAllocate, unsigned long numBytesAllocate) = 0;

		virtual void freeMemoryOnTargetDevice(void* arrayToFree) = 0;

		virtual void initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
				V image1PixelsCompDevice, V image2PixelsCompDevice, const dataCostData<U>& dataCostDeviceCheckerboard) = 0;

		virtual void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
				const levelProperties& prevLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const dataCostData<U>& dataCostDeviceCheckerboardWriteTo) = 0;

		virtual void initializeMessageValsToDefault(
				const levelProperties& currentLevelProperties,
				const checkerboardMessages<U>& messagesDeviceCheckerboard0,
				const checkerboardMessages<U>& messagesDeviceCheckerboard1) = 0;

		virtual void runBPAtCurrentLevel(const BPsettings& algSettings,
				const levelProperties& currentLevelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const checkerboardMessages<U>& messagesDeviceCheckerboard0,
				const checkerboardMessages<U>& messagesDeviceCheckerboard1) = 0;

		virtual void copyMessageValuesToNextLevelDown(
				const levelProperties& currentLevelProperties,
				const levelProperties& nextlevelProperties,
				const checkerboardMessages<U>& messagesDeviceCheckerboard0CopyFrom,
				const checkerboardMessages<U>& messagesDeviceCheckerboard1CopyFrom,
				const checkerboardMessages<U>& messagesDeviceCheckerboard0CopyTo,
				const checkerboardMessages<U>& messagesDeviceCheckerboard1CopyTo) = 0;

		virtual V retrieveOutputDisparity(
				const Checkerboard_Parts currentCheckerboardSet,
				const levelProperties& levelProperties,
				const dataCostData<U>& dataCostDeviceCheckerboard,
				const checkerboardMessages<U>& messagesDeviceSet0Checkerboard0,
				const checkerboardMessages<U>& messagesDeviceSet0Checkerboard1,
				const checkerboardMessages<U>& messagesDeviceSet1Checkerboard0,
				const checkerboardMessages<U>& messagesDeviceSet1Checkerboard1) = 0;

		virtual int getPaddedCheckerboardWidth(int checkerboardWidth);

		unsigned long getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements);

		virtual void freeCheckerboardMessagesMemory(const checkerboardMessages<U>& checkerboardMessagesToFree)
		{
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesU);
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesD);
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesL);
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesR);
		}

		virtual checkerboardMessages<U> allocateMemoryForCheckerboardMessages(unsigned long numDataAllocatePerMessage)
		{
			checkerboardMessages<U> outputCheckerboardMessages;

			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesU, numDataAllocatePerMessage * sizeof(T));
			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesD, numDataAllocatePerMessage * sizeof(T));
			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesL, numDataAllocatePerMessage * sizeof(T));
			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesR, numDataAllocatePerMessage * sizeof(T));

			return outputCheckerboardMessages;
		}

		virtual checkerboardMessages<U> retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(const checkerboardMessages<U>& allCheckerboardMessages, unsigned long offsetIntoAllCheckerboardMessages)
		{
			checkerboardMessages<U> outputCheckerboardMessages;

			outputCheckerboardMessages.messagesU = &(allCheckerboardMessages.messagesU[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesD = &(allCheckerboardMessages.messagesD[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesL = &(allCheckerboardMessages.messagesL[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesR = &(allCheckerboardMessages.messagesR[offsetIntoAllCheckerboardMessages]);

			return outputCheckerboardMessages;
		}

		virtual void freeDataCostsMemory(const dataCostData<U>& dataCostsToFree)
		{
			freeMemoryOnTargetDevice((void*)dataCostsToFree.dataCostCheckerboard0);
			freeMemoryOnTargetDevice((void*)dataCostsToFree.dataCostCheckerboard1);
		}

		virtual dataCostData<U> allocateMemoryForDataCosts(unsigned long numDataAllocatePerDataCostsCheckerboard)
		{
			dataCostData<U> outputDataCosts;

			allocateMemoryOnTargetDevice((void**)&outputDataCosts.dataCostCheckerboard0, numDataAllocatePerDataCostsCheckerboard * sizeof(T));
			allocateMemoryOnTargetDevice((void**)&outputDataCosts.dataCostCheckerboard1, numDataAllocatePerDataCostsCheckerboard * sizeof(T));

			return outputDataCosts;
		}

		virtual std::tuple<dataCostData<U>, checkerboardMessages<U>, checkerboardMessages<U>> allocateAndOrganizeDataCostsAndMessageDataAllLevels(unsigned long numDataAllocatePerDataCostsMessageDataArray)
		{
			dataCostData<U> dataCostsDeviceCheckerboardAllLevels;
			checkerboardMessages<U> messagesDeviceCheckerboard0AllLevels;
			checkerboardMessages<U> messagesDeviceCheckerboard1AllLevels;

			allocateMemoryOnTargetDevice((void**)&dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0, 10*numDataAllocatePerDataCostsMessageDataArray*sizeof(T));
			dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard1 = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[1 * (numDataAllocatePerDataCostsMessageDataArray)]);

			messagesDeviceCheckerboard0AllLevels.messagesU = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[2 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceCheckerboard0AllLevels.messagesD = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[3 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceCheckerboard0AllLevels.messagesL = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[4 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceCheckerboard0AllLevels.messagesR = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[5 * (numDataAllocatePerDataCostsMessageDataArray)]);

			messagesDeviceCheckerboard1AllLevels.messagesU = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[6 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceCheckerboard1AllLevels.messagesD = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[7 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceCheckerboard1AllLevels.messagesL = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[8 * (numDataAllocatePerDataCostsMessageDataArray)]);
			messagesDeviceCheckerboard1AllLevels.messagesR = &(dataCostsDeviceCheckerboardAllLevels.dataCostCheckerboard0[9 * (numDataAllocatePerDataCostsMessageDataArray)]);

			return std::make_tuple(dataCostsDeviceCheckerboardAllLevels, messagesDeviceCheckerboard0AllLevels, messagesDeviceCheckerboard1AllLevels);
		}

		virtual void freeDataCostsAllDataInSingleArray(const dataCostData<U>& dataCostsToFree)
		{
			freeMemoryOnTargetDevice((void*)dataCostsToFree.dataCostCheckerboard0);
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
		std::pair<V, DetailedTimings<Runtime_Type_BP>> operator()(V image1PixelsCompDevice,
			V image2PixelsCompDevice, const BPsettings& algSettings, unsigned int widthImages, unsigned int heightImages);
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
