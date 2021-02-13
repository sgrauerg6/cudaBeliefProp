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

//used code from https://github.com/microsoft/DirectXMath/blob/master/Extensions/DirectXMathF16C.h
//for the values conversion on Windows since _cvtsh_ss and _cvtss_sh not supported in Visual Studio
template<> inline
short getZeroVal<short>()
{
#ifdef _WIN32
	__m128 dataInAvxReg = _mm_set_ss(0.0);
	__m128i convertedData = _mm_cvtps_ph(dataInAvxReg, 0);
	return ((short*)& convertedData)[0];
#else
	return _cvtss_sh(0.0f, 0);
#endif
}

template<> inline
float convertValToDifferentDataTypeIfNeeded<short, float>(const short data)
{
#ifdef _WIN32
	__m128i dataInAvxReg = _mm_cvtsi32_si128(static_cast<int>(data));
	__m128 convertedData = _mm_cvtph_ps(dataInAvxReg);
	return ((float*)& convertedData)[0];
#else
	return _cvtsh_ss(data);
#endif
}

template<> inline
short convertValToDifferentDataTypeIfNeeded<float, short>(const float data)
{
#ifdef _WIN32
	__m128 dataInAvxReg = _mm_set_ss(data);
	__m128i convertedData = _mm_cvtps_ph(dataInAvxReg, 0);
	return ((short*)&convertedData)[0];
#else
	return _cvtss_sh(data, 0);
#endif
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum)
{
	initializeCurrentLevelDataStereoPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
