/*
 * KernelBpStereoCPU_AVX256TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
		float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
		float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{8u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float, __m256, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{8u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<short, __m128i, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
		double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
		double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<double, __m256d, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline __m256d KernelBpStereoCPU::loadPackedDataAligned<double, __m256d>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData) {
	return _mm256_load_pd(
		&inData[retrieveIndexInDataAndMessage(x, y,
			currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity,
			numDispVals)]);
}

template<> inline __m256 KernelBpStereoCPU::loadPackedDataAligned<float, __m256>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return _mm256_load_ps(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline __m128i KernelBpStereoCPU::loadPackedDataAligned<short, __m128i>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData) {
	return _mm_load_si128(
			(__m128i *) &inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline __m256 KernelBpStereoCPU::loadPackedDataUnaligned<float, __m256>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return _mm256_loadu_ps(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline __m128i KernelBpStereoCPU::loadPackedDataUnaligned<short, __m128i>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData) {
	return _mm_loadu_si128(
			(__m128i *) &inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline __m256d KernelBpStereoCPU::loadPackedDataUnaligned<double, __m256d>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData) {
	return _mm256_loadu_pd(
		&inData[retrieveIndexInDataAndMessage(x, y,
			currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity,
			numDispVals)]);
}

template<> inline
__m256 KernelBpStereoCPU::createSIMDVectorSameData<__m256>(const float data) {
	return _mm256_set1_ps(data);
}

template<> inline
__m128i KernelBpStereoCPU::createSIMDVectorSameData<__m128i>(const float data) {
	return _mm256_cvtps_ph(_mm256_set1_ps(data), 0);
}

template<> inline
__m256d KernelBpStereoCPU::createSIMDVectorSameData<__m256d>(const float data) {
	return _mm256_set1_pd(data);
}

#endif /* KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_ */
