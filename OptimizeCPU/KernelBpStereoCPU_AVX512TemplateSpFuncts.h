/*
 * KernelBpStereoCPU_AVX512TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
		float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
		float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{16u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float, __m512, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{16u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<short, __m256i, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline __m512 KernelBpStereoCPU::loadPackedDataAligned<float, __m512>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return _mm512_load_ps(
			&inData[retrieveIndexInDataAndMessage(x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m256i KernelBpStereoCPU::loadPackedDataAligned<short, __m256i>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData) {
	return _mm256_load_si256(
			(__m256i *) &inData[retrieveIndexInDataAndMessage(x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m512 KernelBpStereoCPU::loadPackedDataUnaligned<float, __m512>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return _mm512_loadu_ps(
			&inData[retrieveIndexInDataAndMessage(x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m256i KernelBpStereoCPU::loadPackedDataUnaligned<short, __m256i>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData) {
	return _mm256_loadu_si256(
			(__m256i *) &inData[retrieveIndexInDataAndMessage(x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m512 KernelBpStereoCPU::createSIMDVectorSameData<__m512>(const float data) {
	return _mm512_set1_ps(data);
}

template<> inline __m256i KernelBpStereoCPU::createSIMDVectorSameData<__m256i>(const float data) {
	return _mm512_cvtps_ph(_mm512_set1_ps(data), 0);
}

#endif /* KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_ */
