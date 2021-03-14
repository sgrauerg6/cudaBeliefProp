/*
 * KernelBpStereoCPU_NEON.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_NEON_H_
#define KERNELBPSTEREOCPU_NEON_H_

//this is only used when processing using an ARM CPU with NEON instructions
#include <arm_neon.h>

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
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float, float32x4_t, DISP_VALS>(
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
		float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
		float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
		float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
		float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
		float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int bpSettingsDispVals)
{
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float16_t, float16x4_t, DISP_VALS>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector, bpSettingsDispVals);
}

/*template<unsigned int DISP_VALS>
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
}*/

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataAligned<float, float32x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return vld1q_f32(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataAligned<float16_t, float16x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float16_t* inData) {
	return vld1_f16(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity,
			numDispVals)]);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float, float32x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return vld1q_f32(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float16_t, float16x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float16_t* inData) {
	return vld1_f16(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}

/*template<> inline __m256d KernelBpStereoCPU::loadPackedDataUnaligned<double, __m256d>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, double* inData) {
	return _mm256_loadu_pd(&inData[retrieveIndexInDataAndMessage(
			x, y, currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, currentDisparity, numDispVals)]);
}*/

template<> inline float32x4_t KernelBpStereoCPU::createSIMDVectorSameData<float32x4_t>(const float data) {
	return vdupq_n_f32(data);
}

template<> inline float16x4_t KernelBpStereoCPU::createSIMDVectorSameData<float16x4_t>(const float data) {
	return vcvt_f16_f32(createSIMDVectorSameData<float32x4_t>(data));
}

/*template<> inline __m256d KernelBpStereoCPU::createSIMDVectorSameData<__m256d>(const float data) {
	return _mm256_set1_pd(data);
}*/

template<> inline float32x4_t KernelBpStereoCPU::addVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
	return vaddq_f32(val1, val2);
}

/*template<> inline __m256d KernelBpStereoCPU::addVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_add_pd(val1, val2);
}*/

template<> inline float32x4_t KernelBpStereoCPU::addVals<float32x4_t, float16x4_t, float32x4_t>(const float32x4_t& val1, const float16x4_t& val2) {
	return vaddq_f32(val1, vcvt_f32_f16(val2));
}

template<> inline float32x4_t KernelBpStereoCPU::addVals<float16x4_t, float32x4_t, float32x4_t>(const float16x4_t& val1, const float32x4_t& val2) {
	return vaddq_f32(vcvt_f32_f16(val1), val2);
}

template<> inline float32x4_t KernelBpStereoCPU::addVals<float16x4_t, float16x4_t, float32x4_t>(const float16x4_t& val1, const float16x4_t& val2) {
	return vaddq_f32(vcvt_f32_f16(val1), vcvt_f32_f16(val2));
}

template<> inline float32x4_t KernelBpStereoCPU::subtractVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
	return vsubq_f32(val1, val2);
}

/*template<> inline __m256d KernelBpStereoCPU::subtractVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_sub_pd(val1, val2);
}*/

template<> inline float32x4_t KernelBpStereoCPU::divideVals<float32x4_t, float32x4_t, float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
	return vdivq_f32(val1, val2);
}

/*template<> inline __m256d KernelBpStereoCPU::divideVals<__m256d, __m256d, __m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_div_pd(val1, val2);
}*/

template<> inline float32x4_t KernelBpStereoCPU::convertValToDatatype<float32x4_t, float>(const float val) {
	return vdupq_n_f32(val);
}

/*template<> inline __m256d KernelBpStereoCPU::convertValToDatatype<__m256d, double>(const double val) {
	return _mm256_set1_pd(val);
}*/

template<> inline float32x4_t KernelBpStereoCPU::getMinByElement<float32x4_t>(const float32x4_t& val1, const float32x4_t& val2) {
	return vminnmq_f32(val1, val2);
}

/*template<> inline __m256d KernelBpStereoCPU::getMinByElement<__m256d>(const __m256d& val1, const __m256d& val2) {
	return _mm256_min_pd(val1, val2);
}*/

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float, float32x4_t>(
		const unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore) {
	vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float16_t, float32x4_t>(
		const unsigned int indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore) {
	vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

/*template<> inline void KernelBpStereoCPU::storePackedDataAligned<double, __m256d>(
		const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
	_mm256_store_pd(&locationDataStore[indexDataStore], dataToStore);
}*/

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float, float32x4_t>(
		const unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore) {
	vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float16_t, float32x4_t>(
		const unsigned int indexDataStore, float16_t* locationDataStore, const float32x4_t& dataToStore) {
	vst1_f16(&locationDataStore[indexDataStore], vcvt_f16_f32(dataToStore));
}

/*template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<double, __m256d>(
		const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
	_mm256_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
}*/

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(
			xVal, yVal, currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

template<> inline void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t>(const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		float16x4_t* messageValsNeighbor1, float16x4_t* messageValsNeighbor2,
		float16x4_t* messageValsNeighbor3, float16x4_t* dataCosts,
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned,
		const unsigned int bpSettingsDispVals)
{
	msgStereoSIMDProcessing<float16_t, float16x4_t, float, float32x4_t>(xVal, yVal, currentLevelProperties,
			messageValsNeighbor1, messageValsNeighbor2, messageValsNeighbor3, dataCosts,
			dstMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}

/*template<unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
		float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
		float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp) {
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float, float32x4_t, DISP_VALS>(
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
		float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
		float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
		float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
		float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
		float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp) {
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<
			float16_t, float16x4_t, DISP_VALS>(checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataAligned<float, float32x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return vld1q_f32(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataAligned<float16_t, float16x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float16_t* inData) {
	return vld1_f16(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float, float32x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float* inData) {
	return vld1q_f32(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float16_t, float16x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, const unsigned int numDispVals, float16_t* inData) {
	return vld1_f16(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					numDispVals)]);
}

template<> inline float32x4_t KernelBpStereoCPU::createSIMDVectorSameData<float32x4_t>(const float data) {
	return vdupq_n_f32(data);
}

template<> inline float16x4_t KernelBpStereoCPU::createSIMDVectorSameData<float16x4_t>(const float data) {
	return vcvt_f16_f32(createSIMDVectorSameData<float32x4_t>(data));
}*/

#endif /* KERNELBPSTEREOCPU_NEON_H_ */
