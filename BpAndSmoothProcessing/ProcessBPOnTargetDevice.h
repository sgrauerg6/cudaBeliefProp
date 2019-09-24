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

		virtual void initializeDataCosts(const BPsettings& algSettings, levelProperties& currentLevelProperties,
				float* image1PixelsCompDevice, float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
				T* dataCostDeviceCheckerboard2) = 0;

		virtual void initializeDataCurrentLevel(levelProperties& currentLevelPropertes,
				levelProperties& prevLevelProperties,
				T* dataCostStereoCheckerboard1,
				T* dataCostStereoCheckerboard2,
				T* dataCostDeviceToWriteToCheckerboard1,
				T* dataCostDeviceToWriteToCheckerboard2) = 0;

		virtual void initializeMessageValsToDefault(
				levelProperties& currentLevelPropertes,
				T* messageUDeviceCheckerboard1,
				T* messageDDeviceCheckerboard1,
				T* messageLDeviceCheckerboard1,
				T* messageRDeviceCheckerboard1,
				T* messageUDeviceCheckerboard2,
				T* messageDDeviceCheckerboard2,
				T* messageLDeviceCheckerboard2,
				T* messageRDeviceCheckerboard2) = 0;

		virtual void runBPAtCurrentLevel(const BPsettings& algSettings,
				levelProperties& currentLevelPropertes,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2,
				T* messageUDeviceCheckerboard1,
				T* messageDDeviceCheckerboard1,
				T* messageLDeviceCheckerboard1,
				T* messageRDeviceCheckerboard1,
				T* messageUDeviceCheckerboard2,
				T* messageDDeviceCheckerboard2,
				T* messageLDeviceCheckerboard2,
				T* messageRDeviceCheckerboard2) = 0;

		virtual void copyMessageValuesToNextLevelDown(
				levelProperties& currentLevelPropertes,
				levelProperties& nextLevelPropertes,
				T* messageUDeviceCheckerboard1CopyFrom,
				T* messageDDeviceCheckerboard1CopyFrom,
				T* messageLDeviceCheckerboard1CopyFrom,
				T* messageRDeviceCheckerboard1CopyFrom,
				T* messageUDeviceCheckerboard2CopyFrom,
				T* messageDDeviceCheckerboard2CopyFrom,
				T* messageLDeviceCheckerboard2CopyFrom,
				T* messageRDeviceCheckerboard2CopyFrom,
				T* messageUDeviceCheckerboard1CopyTo,
				T* messageDDeviceCheckerboard1CopyTo,
				T* messageLDeviceCheckerboard1CopyTo,
				T* messageRDeviceCheckerboard1CopyTo,
				T* messageUDeviceCheckerboard2CopyTo,
				T* messageDDeviceCheckerboard2CopyTo,
				T* messageLDeviceCheckerboard2CopyTo,
				T* messageRDeviceCheckerboard2CopyTo) = 0;

		virtual void retrieveOutputDisparity(
				int currentCheckerboardSet,
				levelProperties& levelPropertes,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2,
				T* messageUDeviceSet0Checkerboard1,
				T* messageDDeviceSet0Checkerboard1,
				T* messageLDeviceSet0Checkerboard1,
				T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2,
				T* messageDDeviceSet0Checkerboard2,
				T* messageLDeviceSet0Checkerboard2,
				T* messageRDeviceSet0Checkerboard2,
				T* messageUDeviceSet1Checkerboard1,
				T* messageDDeviceSet1Checkerboard1,
				T* messageLDeviceSet1Checkerboard1,
				T* messageRDeviceSet1Checkerboard1,
				T* messageUDeviceSet1Checkerboard2,
				T* messageDDeviceSet1Checkerboard2,
				T* messageLDeviceSet1Checkerboard2,
				T* messageRDeviceSet1Checkerboard2,
				float* resultingDisparityMapCompDevice) = 0;

		virtual int getPaddedCheckerboardWidth(int checkerboardWidth);

		unsigned long getNumDataForAlignedMemoryAtLevel(unsigned int widthLevelActualIntegerSize, unsigned int heightLevelActualIntegerSize, unsigned int totalPossibleMovements);

		//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
		//input is images image1Pixels and image1Pixels
		//output is resultingDisparityMap
		DetailedTimings<Runtime_Type_BP> operator()(float* image1PixelsCompDevice,
			float* image2PixelsCompDevice,
			float* resultingDisparityMapCompDevice, const BPsettings& algSettings, unsigned int widthImages, unsigned int heightImages);
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
