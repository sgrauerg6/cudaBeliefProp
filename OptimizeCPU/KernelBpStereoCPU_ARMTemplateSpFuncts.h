/*
 * KernelBpStereoCPU_ARMTemplateSpFuncts.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_


#include "KernelBpStereoCPU.h"
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

template<> inline
float16_t getZeroVal<float16_t>()
{
	return (float16_t)0.0f;
}

template<> inline
float convertValToDifferentDataTypeIfNeeded<float16_t, float>(float16_t valToConvert)
{
	return (float)valToConvert;
}

template<> inline
float16_t convertValToDifferentDataTypeIfNeeded<float, float16_t>(float valToConvert)
{
	//seems like simple cast function works
	return (float16_t)valToConvert;
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float16_t>(int xVal, int yVal, int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2,
		float16_t* messageUDeviceCurrentCheckerboard1,
		float16_t* messageDDeviceCurrentCheckerboard1,
		float16_t* messageLDeviceCurrentCheckerboard1,
		float16_t* messageRDeviceCurrentCheckerboard1,
		float16_t* messageUDeviceCurrentCheckerboard2,
		float16_t* messageDDeviceCurrentCheckerboard2,
		float16_t* messageLDeviceCurrentCheckerboard2,
		float16_t* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
		int offsetData, bool dataAligned)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float16_t, float>(
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
				offsetData, dataAligned);
}

template<> inline
void initializeBottomLevelDataStereoPixel<float16_t, float16_t>(int xVal, int yVal, levelProperties& currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, float16_t* dataCostDeviceStereoCheckerboard1, float16_t* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp)
{
	initializeBottomLevelDataStereoPixel<float16_t, float>(xVal, yVal,
			currentLevelProperties, image1PixelsDevice,
			image2PixelsDevice, dataCostDeviceStereoCheckerboard1,
			dataCostDeviceStereoCheckerboard2, lambda_bp,
			data_k_bp);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<float16_t, float16_t>(int xVal, int yVal, int checkerboardPart, levelProperties& currentLevelProperties, levelProperties& prevLevelProperties, float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2, float16_t* dataCostDeviceToWriteTo, int offsetNum)
{
	initializeCurrentLevelDataStereoPixel<float16_t, float>(
			xVal, yVal, checkerboardPart,
			currentLevelProperties,
			prevLevelProperties, dataCostStereoCheckerboard1,
			dataCostStereoCheckerboard2, dataCostDeviceToWriteTo,
			offsetNum);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float16_t>(int xVal, int yVal, levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2, float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1, float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1, float16_t* messageUPrevStereoCheckerboard2, float16_t* messageDPrevStereoCheckerboard2, float16_t* messageLPrevStereoCheckerboard2, float16_t* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<float16_t, float>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard1, dataCostStereoCheckerboard2, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1, messageUPrevStereoCheckerboard2, messageDPrevStereoCheckerboard2, messageLPrevStereoCheckerboard2, messageRPrevStereoCheckerboard2, disparityBetweenImagesDevice);
}

#endif



#endif /* KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_ */
