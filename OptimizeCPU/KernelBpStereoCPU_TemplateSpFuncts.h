#include "KernelBpStereoCPU.h"

#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS

//this is only processed when on x86
#include <x86intrin.h>
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

template<> inline
short getZeroVal<short>()
{
	return _cvtss_sh(0.0f, 0);
}

template<> inline
float convertValToDifferentDataTypeIfNeeded<short, float>(short data)
{
	return _cvtsh_ss(data);
}

template<> inline
short convertValToDifferentDataTypeIfNeeded<float, short>(float data)
{
	return _cvtss_sh(data, 0);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short>(int xVal, int yVal, int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2,
		short* messageUDeviceCurrentCheckerboard1,
		short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1,
		short* messageUDeviceCurrentCheckerboard2,
		short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2,
		short* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
		int offsetData)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float>(
			xVal, yVal, checkerboardToUpdate,
			currentLevelProperties,
			dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1,
			messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1,
			messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2,
			messageDDeviceCurrentCheckerboard2,
			messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, disc_k_bp,
			offsetData);
}



template<> inline
void initializeBottomLevelDataStereoPixel<short, short>(int xVal, int yVal, levelProperties& currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, short* dataCostDeviceStereoCheckerboard1, short* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp)
{
		initializeBottomLevelDataStereoPixel<short, float>(xVal, yVal,
				currentLevelProperties, image1PixelsDevice,
				image2PixelsDevice, dataCostDeviceStereoCheckerboard1,
				dataCostDeviceStereoCheckerboard2, lambda_bp,
				data_k_bp);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoNoTexturesPixel<short, short>(int xVal, int yVal, int checkerboardPart, levelProperties& currentLevelProperties, levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* dataCostDeviceToWriteTo, int offsetNum)
{
	initializeCurrentLevelDataStereoNoTexturesPixel<short, float>(
					xVal, yVal, checkerboardPart,
					currentLevelProperties,
					prevLevelProperties, dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2, dataCostDeviceToWriteTo,
					offsetNum);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short>(int xVal, int yVal, levelProperties& currentLevelProperties, short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1, short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, short* messageUPrevStereoCheckerboard2, short* messageDPrevStereoCheckerboard2, short* messageLPrevStereoCheckerboard2, short* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1, messageUPrevStereoCheckerboard2, messageDPrevStereoCheckerboard2, messageLPrevStereoCheckerboard2, messageRPrevStereoCheckerboard2, disparityBetweenImagesDevice);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
