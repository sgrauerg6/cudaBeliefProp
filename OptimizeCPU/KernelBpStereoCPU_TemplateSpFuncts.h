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
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, 0>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, 0>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, 0>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals,
		void* dstProcessing)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, 0>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals, dstProcessing);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

template<> inline
void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
			xVal, yVal, checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, offsetData, dataAligned, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, 0>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	initializeCurrentLevelDataStereoPixel<short, float, 0>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	initializeCurrentLevelDataStereoPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	initializeCurrentLevelDataStereoPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	initializeCurrentLevelDataStereoPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	initializeCurrentLevelDataStereoPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void initializeCurrentLevelDataStereoPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties, short* dataCostStereoCheckerboard0,
		short* dataCostStereoCheckerboard1, short* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	initializeCurrentLevelDataStereoPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(xVal, yVal, checkerboardPart, currentLevelProperties, prevLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1, dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, 0>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, 0>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

template<> inline
void retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1, short* messageUPrevStereoCheckerboard0, short* messageDPrevStereoCheckerboard0,
		short* messageLPrevStereoCheckerboard0, short* messageRPrevStereoCheckerboard0, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1,
		short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	retrieveOutputDisparityCheckerboardStereoOptimizedPixel<short, float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(xVal, yVal, currentLevelProperties, dataCostStereoCheckerboard0,
			dataCostStereoCheckerboard1, messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0, messageLPrevStereoCheckerboard0,
			messageRPrevStereoCheckerboard0, messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1, messageLPrevStereoCheckerboard1,
			messageRPrevStereoCheckerboard1, disparityBetweenImagesDevice, bpSettingsDispVals);
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
