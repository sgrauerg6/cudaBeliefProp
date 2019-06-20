/*
 * ProcessBPOnTargetDeviceHelperFuncts.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICEHELPERFUNCTS_H_
#define PROCESSBPONTARGETDEVICEHELPERFUNCTS_H_

#include "bpStereoParameters.h"

template<typename T>
class ProcessBPOnTargetDeviceHelperFuncts {
public:
	ProcessBPOnTargetDeviceHelperFuncts();
	virtual ~ProcessBPOnTargetDeviceHelperFuncts();

	virtual int getCheckerboardWidthCPU(int widthLevelActualIntegerSize);

	virtual void allocateMemoryOnTargetDevice(void** arrayToAllocate, int numBytesAllocate);

	virtual void freeMemoryOnTargetDevice(void* arrayToFree);

	virtual void initialDataCosts(float* image1PixelsCompDevice,
			float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
			T* dataCostDeviceCheckerboard2, BPsettings& algSettings);

	virtual void initializeDataCurrentLevel(T* dataCostStereoCheckerboard1,
			T* dataCostStereoCheckerboard2,
			T* dataCostDeviceToWriteToCheckerboard1,
			T* dataCostDeviceToWriteToCheckerboard2,
			int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
			int prevWidthLevelActualIntegerSize,
			int prevHeightLevelActualIntegerSize);

	virtual void initializeMessageValsToDefault(
			T* messageUDeviceSet0Checkerboard1,
			T* messageDDeviceSet0Checkerboard1,
			T* messageLDeviceSet0Checkerboard1,
			T* messageRDeviceSet0Checkerboard1,
			T* messageUDeviceSet0Checkerboard2,
			T* messageDDeviceSet0Checkerboard2,
			T* messageLDeviceSet0Checkerboard2,
			T* messageRDeviceSet0Checkerboard2, int widthLevelActualIntegerSize,
			int heightLevelActualIntegerSize, int totalPossibleMovements);

	virtual void runBPAtCurrentLevel(BPsettings& algSettings,
			int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
			T* messageUDeviceSet0Checkerboard1,
			T* messageDDeviceSet0Checkerboard1,
			T* messageLDeviceSet0Checkerboard1,
			T* messageRDeviceSet0Checkerboard1,
			T* messageUDeviceSet0Checkerboard2,
			T* messageDDeviceSet0Checkerboard2,
			T* messageLDeviceSet0Checkerboard2,
			T* messageRDeviceSet0Checkerboard2,
			T* dataCostDeviceCurrentLevelCheckerboard1,
			T* dataCostDeviceCurrentLevelCheckerboard2);

	virtual void copyMessageValuesToNextLevelDown(
			int prevWidthLevelActualIntegerSize,
			int prevHeightLevelActualIntegerSize,
			int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
			T* messageUDeviceSet0Checkerboard1,
			T* messageDDeviceSet0Checkerboard1,
			T* messageLDeviceSet0Checkerboard1,
			T* messageRDeviceSet0Checkerboard1,
			T* messageUDeviceSet0Checkerboard2,
			T* messageDDeviceSet0Checkerboard2,
			T* messageLDeviceSet0Checkerboard2,
			T* messageRDeviceSet0Checkerboard2,
			T** messageUDeviceSet1Checkerboard1,
			T** messageDDeviceSet1Checkerboard1,
			T** messageLDeviceSet1Checkerboard1,
			T** messageRDeviceSet1Checkerboard1,
			T** messageUDeviceSet1Checkerboard2,
			T** messageDDeviceSet1Checkerboard2,
			T** messageLDeviceSet1Checkerboard2,
			T** messageRDeviceSet1Checkerboard2);

	virtual void retrieveOutputDisparity(
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
			float* resultingDisparityMapCompDevice, int widthLevel,
			int heightLevel, int currentCheckerboardSet);
};

#endif /* PROCESSBPONTARGETDEVICEHELPERFUNCTS_H_ */
