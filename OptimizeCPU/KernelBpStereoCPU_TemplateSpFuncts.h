#include "KernelBpStereoCPU.h"

#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS

//this is only processed when on x86
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

template<> inline
short getZeroVal<short>()
{
#ifdef _WIN32
	float dataArray[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	__m128i convertedData = _mm256_cvtps_ph(_mm256_load_ps(dataArray), 0);
	return ((short*)& convertedData)[0];
#else
	return _cvtss_sh(0.0f, 0);
#endif
}

template<> inline
float convertValToDifferentDataTypeIfNeeded<short, float>(short data)
{
#ifdef _WIN32
	short dataArray[8] = { data, data, data, data, data, data, data, data };
	__m128i* dataPacked = (__m128i*) & dataArray;
	__m256 convertedData = _mm256_cvtph_ps(dataPacked[0]);
	return ((float*)&convertedData)[0];
#else
	return _cvtsh_ss(data);
#endif
}

template<> inline
short convertValToDifferentDataTypeIfNeeded<float, short>(float data)
{
#ifdef _WIN32
	float dataArray[8] = { data, data, data, data, data, data, data, data };
	__m256* dataPacked = (__m256*) & dataArray;
	__m128i convertedData = _mm256_cvtps_ph(dataPacked[0], 0);
	return ((short*)&convertedData)[0];
#else
	return _cvtss_sh(data, 0);
#endif
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short>(int xVal, int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0,
		short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0,
		short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1,
		short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1, float disc_k_bp,
		int offsetData, bool dataAligned)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float>(
			xVal, yVal, checkerboardToUpdate,
			currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0,
			messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0,
			messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1,
			messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1,
			messageRDeviceCurrentCheckerboard1, disc_k_bp,
			offsetData, dataAligned);
}



template<> inline
void initializeBottomLevelDataStereoPixel<short, short>(int xVal, int yVal, const levelProperties& currentLevelProperties,
		float* image1PixelsDevice, float* image2PixelsDevice, short* dataCostDeviceStereoCheckerboard0,
		short* dataCostDeviceStereoCheckerboard1, float lambda_bp, float data_k_bp)
{
	initializeBottomLevelDataStereoPixel<short, float>(xVal, yVal,
			currentLevelProperties, image1PixelsDevice,
			image2PixelsDevice, dataCostDeviceStereoCheckerboard0,
			dataCostDeviceStereoCheckerboard1, lambda_bp,
			data_k_bp);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short>(int xVal, int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, int offsetNum)
{
	initializeCurrentLevelDataStereoPixel<short, float>(
			xVal, yVal, checkerboardPart,
			currentLevelProperties,
			prevLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, dataCostDeviceToWriteTo,
			offsetNum);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short>(int xVal, int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
