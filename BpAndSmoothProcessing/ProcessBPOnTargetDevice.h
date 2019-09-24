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

template<typename T>
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
				float* image1PixelsCompDevice, float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
				T* dataCostDeviceCheckerboard2) = 0;

		virtual void initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
				const levelProperties& prevLevelProperties,
				T* dataCostStereoCheckerboard1,
				T* dataCostStereoCheckerboard2,
				T* dataCostDeviceToWriteToCheckerboard1,
				T* dataCostDeviceToWriteToCheckerboard2) = 0;

		virtual void initializeMessageValsToDefault(
				const levelProperties& currentLevelProperties,
				const checkerboardMessages<T>& messagesDeviceCheckerboard0,
				const checkerboardMessages<T>& messagesDeviceCheckerboard1) = 0;

		virtual void runBPAtCurrentLevel(const BPsettings& algSettings,
				const levelProperties& currentLevelProperties,
				T* dataCostDeviceCurrentLevelCheckerboard0,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				const checkerboardMessages<T>& messagesDeviceCheckerboard0,
				const checkerboardMessages<T>& messagesDeviceCheckerboard1) = 0;

		virtual void copyMessageValuesToNextLevelDown(
				const levelProperties& currentLevelProperties,
				const levelProperties& nextlevelProperties,
				const checkerboardMessages<T>& messagesDeviceCheckerboard0CopyFrom,
				const checkerboardMessages<T>& messagesDeviceCheckerboard1CopyFrom,
				const checkerboardMessages<T>& messagesDeviceCheckerboard0CopyTo,
				const checkerboardMessages<T>& messagesDeviceCheckerboard1CopyTo) = 0;

		virtual void retrieveOutputDisparity(
				const Checkerboard_Parts currentCheckerboardSet,
				const levelProperties& levelProperties,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2,
				const checkerboardMessages<T>& messagesDeviceSet0Checkerboard0,
				const checkerboardMessages<T>& messagesDeviceSet0Checkerboard1,
				const checkerboardMessages<T>& messagesDeviceSet1Checkerboard0,
				const checkerboardMessages<T>& messagesDeviceSet1Checkerboard1,
				float* resultingDisparityMapCompDevice) = 0;

		virtual int getPaddedCheckerboardWidth(int checkerboardWidth);

		unsigned long getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements);

		void freeCheckerboardMessagesMemory(const checkerboardMessages<T>& checkerboardMessagesToFree)
		{
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesU);
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesD);
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesL);
			freeMemoryOnTargetDevice((void*)checkerboardMessagesToFree.messagesR);
		}

		checkerboardMessages<T> allocateMemoryForCheckerboardMessages(unsigned long numBytesAllocatePerMessage)
		{
			checkerboardMessages<T> outputCheckerboardMessages;

			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesU, numBytesAllocatePerMessage);
			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesD, numBytesAllocatePerMessage);
			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesL, numBytesAllocatePerMessage);
			allocateMemoryOnTargetDevice((void**)&outputCheckerboardMessages.messagesR, numBytesAllocatePerMessage);

			return outputCheckerboardMessages;
		}

		checkerboardMessages<T> retrieveCurrentCheckerboardMessagesFromOffsetIntoAllCheckerboardMessages(const checkerboardMessages<T>& allCheckerboardMessages, unsigned long offsetIntoAllCheckerboardMessages)
		{
			checkerboardMessages<T> outputCheckerboardMessages;

			outputCheckerboardMessages.messagesU = &(allCheckerboardMessages.messagesU[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesD = &(allCheckerboardMessages.messagesD[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesL = &(allCheckerboardMessages.messagesL[offsetIntoAllCheckerboardMessages]);
			outputCheckerboardMessages.messagesR = &(allCheckerboardMessages.messagesR[offsetIntoAllCheckerboardMessages]);

			return outputCheckerboardMessages;
		}

		//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
		//input is images image1Pixels and image1Pixels
		//output is resultingDisparityMap
		DetailedTimings<Runtime_Type_BP> operator()(float* image1PixelsCompDevice,
			float* image2PixelsCompDevice,
			float* resultingDisparityMapCompDevice, const BPsettings& algSettings, unsigned int widthImages, unsigned int heightImages);
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
