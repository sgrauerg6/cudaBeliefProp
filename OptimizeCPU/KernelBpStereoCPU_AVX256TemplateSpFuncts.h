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
		const float disc_k_bp, const unsigned int bpSettingsDispVals)
{
	constexpr unsigned int numDataInSIMDVector{8u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float, __m256, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector, bpSettingsDispVals);
}

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int bpSettingsDispVals)
{
	constexpr unsigned int numDataInSIMDVector{8u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<short, __m128i, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector, bpSettingsDispVals);
}

template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
		double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
		double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int bpSettingsDispVals)
{
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<double, __m256d, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector, bpSettingsDispVals);
}

template<> inline __m256d KernelBpStereoCPU::loadPackedDataAligned<double, __m256d>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData) {
	return _mm256_load_pd(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m256 KernelBpStereoCPU::loadPackedDataAligned<float, __m256>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return _mm256_load_ps(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m128i KernelBpStereoCPU::loadPackedDataAligned<short, __m128i>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData) {
	return _mm_load_si128((__m128i *)(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity,
			numDispVals)]));
}

template<> inline __m256 KernelBpStereoCPU::loadPackedDataUnaligned<float, __m256>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return _mm256_loadu_ps(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m128i KernelBpStereoCPU::loadPackedDataUnaligned<short, __m128i>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, short* inData) {
	return _mm_loadu_si128((__m128i*)(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]));
}

template<> inline __m256d KernelBpStereoCPU::loadPackedDataUnaligned<double, __m256d>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData) {
	return _mm256_loadu_pd(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline __m256 KernelBpStereoCPU::createSIMDVectorSameData<__m256>(const float data) {
	return _mm256_set1_ps(data);
}

template<> inline __m128i KernelBpStereoCPU::createSIMDVectorSameData<__m128i>(const float data) {
	return _mm256_cvtps_ph(_mm256_set1_ps(data), 0);
}

template<> inline __m256d KernelBpStereoCPU::createSIMDVectorSameData<__m256d>(const float data) {
	return _mm256_set1_pd(data);
}

template<> inline __m256 KernelBpStereoCPU::addVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
	return _mm256_add_ps(val1, val2);
}

template<> inline __m256d KernelBpStereoCPU::addVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_add_pd(val1, val2);
}

template<> inline __m256 KernelBpStereoCPU::addVals<__m256, __m128i, __m256>(const __m256& val1, const __m128i& val2) {
	return _mm256_add_ps(val1, _mm256_cvtph_ps(val2));
}

template<> inline __m256 KernelBpStereoCPU::addVals<__m128i, __m256, __m256>(const __m128i& val1, const __m256& val2) {
	return _mm256_add_ps(_mm256_cvtph_ps(val1), val2);
}

template<> inline __m256 KernelBpStereoCPU::addVals<__m128i, __m128i, __m256>(const __m128i& val1, const __m128i& val2) {
	return _mm256_add_ps(_mm256_cvtph_ps(val1), _mm256_cvtph_ps(val2));
}

template<> inline __m256 KernelBpStereoCPU::subtractVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
	return _mm256_sub_ps(val1, val2);
}

template<> inline __m256d KernelBpStereoCPU::subtractVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_sub_pd(val1, val2);
}

template<> inline __m256 KernelBpStereoCPU::divideVals<__m256, __m256, __m256>(const __m256& val1, const __m256& val2) {
	return _mm256_div_ps(val1, val2);
}

template<> inline __m256d KernelBpStereoCPU::divideVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_div_pd(val1, val2);
}

template<> inline __m256 KernelBpStereoCPU::convertValToDatatype<__m256, float>(const float val) {
	return _mm256_set1_ps(val);
}

template<> inline __m256d KernelBpStereoCPU::convertValToDatatype<__m256d, double>(const double val) {
	return _mm256_set1_pd(val);
}

template<> inline __m256 KernelBpStereoCPU::getMinByElement<__m256>(const __m256& val1, const __m256& val2) {
	return _mm256_min_ps(val1, val2);
}

template<> inline __m256d KernelBpStereoCPU::getMinByElement<__m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_min_pd(val1, val2);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float, __m256>(
		const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore) {
	_mm256_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<short, __m256>(
		const unsigned int indexDataStore, short* locationDataStore, const __m256& dataToStore) {
	_mm_store_si128((__m128i*)(&locationDataStore[indexDataStore]), _mm256_cvtps_ph(dataToStore, 0));
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<double, __m256d>(
		const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
	_mm256_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float, __m256>(
		const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore) {
	_mm256_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<short, __m256>(
		const unsigned int indexDataStore, short* locationDataStore, const __m256& dataToStore) {
	_mm_storeu_si128((__m128i*)(&locationDataStore[indexDataStore]), _mm256_cvtps_ph(dataToStore, 0));
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<double, __m256d>(
		const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
	_mm256_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<> inline void KernelBpStereoCPU::msgStereoSIMD<short, __m128i>(const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i* messageValsNeighbor1, __m128i* messageValsNeighbor2,
		__m128i* messageValsNeighbor3, __m128i* dataCosts,
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned,
		const unsigned int bpSettingsDispVals)
{
	msgStereoSIMDProcessing<short, __m128i, float, __m256>(xVal, yVal, currentLevelProperties,
			messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, dataCosts,
			dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}

#endif /* KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_ */
