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

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
		float>(const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
		float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
		float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp) {
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<
			float, float32x4_t>(checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
float16_t>(const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		float16_t* dataCostStereoCheckerboard0, float16_t* dataCostStereoCheckerboard1,
		float16_t* messageUDeviceCurrentCheckerboard0, float16_t* messageDDeviceCurrentCheckerboard0,
		float16_t* messageLDeviceCurrentCheckerboard0, float16_t* messageRDeviceCurrentCheckerboard0,
		float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1,
		float16_t* messageLDeviceCurrentCheckerboard1, float16_t* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp) {
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<
			float16_t, float16x4_t>(checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataAligned<float,float32x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, float* inData) {
	return vld1q_f32(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataAligned<float16_t, float16x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, float16_t* inData) {
	return vld1_f16(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float, float32x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, float* inData) {
	return vld1q_f32(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float16_t, float16x4_t>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, float16_t* inData) {
	return vld1_f16(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float, float32x4_t>(
		const unsigned int indexDataStore, float* locationDataStore,
		const float32x4_t& dataToStore) {
	vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float16_t, float16x4_t>(
		const unsigned int indexDataStore, float16_t* locationDataStore, const float16x4_t& dataToStore) {
	vst1_f16(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float, float32x4_t>(
		const unsigned int indexDataStore, float* locationDataStore, const float32x4_t& dataToStore) {
	vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float16_t, float16x4_t>(
		const unsigned int indexDataStore, float16_t* locationDataStore, const float16x4_t& dataToStore) {
	vst1_f16(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline float32x4_t KernelBpStereoCPU::createSIMDVectorSameData<float32x4_t>(const float data) {
	return vdupq_n_f32(data);
}

template<> inline float16x4_t KernelBpStereoCPU::createSIMDVectorSameData<float16x4_t>(const float data) {
	return vcvt_f16_f32(createSIMDVectorSameData<float32x4_t>(data));
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoSIMD<float32x4_t>(float32x4_t f[bp_params::NUM_POSSIBLE_DISPARITY_VALUES])
{
	float32x4_t prev;
	float32x4_t vectorAllOneVal = vdupq_n_f32(1.0f);
	for (unsigned int currentDisparity = 1; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = vaddq_f32(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
		 f[currentDisparity] = prev;*/
		f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
	}

	for (int currentDisparity = bp_params::NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = vaddq_f32(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
	}
}

template<> inline
void KernelBpStereoCPU::msgStereoSIMD<float, float32x4_t>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		float32x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		float32x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		float* dstMessageArray, const float32x4_t& disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	//T minimum = INF_BP;
	float32x4_t minimum = vdupq_n_f32(INF_BP);
	float32x4_t dst[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = vaddq_f32(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = vaddq_f32(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = vaddq_f32(dst[currentDisparity], dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = vminnmq_f32(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<float32x4_t>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = vaddq_f32(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	float32x4_t valToNormalize = vdupq_n_f32(0.0f);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = vminnmq_f32(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = vaddq_f32(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)bp_params::NUM_POSSIBLE_DISPARITY_VALUES));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = vsubq_f32(dst[currentDisparity],
				valToNormalize);
		if (dataAligned)
		{
			storePackedDataAligned<float, float32x4_t >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		} else
		{
			storePackedDataUnaligned<float, float32x4_t >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}
		if constexpr (OPTIMIZED_INDEXING_SETTING)
		{
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
		}
		else
		{
			destMessageArrayIndex++;
		}
	}
}

// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoSIMD<float16_t, float16x4_t>(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		float16x4_t messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float16x4_t messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		float16x4_t messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float16x4_t dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	//T minimum = INF_BP;
	float32x4_t minimum = vdupq_n_f32(INF_BP);
	float32x4_t dstFloat[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dstFloat[currentDisparity] = vaddq_f32((vcvt_f32_f16(messageValsNeighbor1[currentDisparity])), (vcvt_f32_f16(messageValsNeighbor2[currentDisparity])));
		dstFloat[currentDisparity] = vaddq_f32(dstFloat[currentDisparity], vcvt_f32_f16(messageValsNeighbor3[currentDisparity]));
		dstFloat[currentDisparity] = vaddq_f32(dstFloat[currentDisparity], vcvt_f32_f16(dataCosts[currentDisparity]));

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = vminnmq_f32(minimum, dstFloat[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<float32x4_t>(dstFloat);

	// truncate
	//minimum += disc_k_bp;
	minimum = vaddq_f32(minimum, vcvt_f32_f16(disc_k_bp));

	// normalize
	//T valToNormalize = 0;
	float32x4_t valToNormalize = vdupq_n_f32(0.0f);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dstFloat[currentDisparity] = vminnmq_f32(minimum, dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = vaddq_f32(valToNormalize, dstFloat[currentDisparity]);
	}

	//valToNormalize /= bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)bp_params::NUM_POSSIBLE_DISPARITY_VALUES));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, 0,
			bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dstFloat[currentDisparity] = vsubq_f32(dstFloat[currentDisparity], valToNormalize);
		if (dataAligned)
		{
			storePackedDataAligned<float16_t, float16x4_t >(destMessageArrayIndex, dstMessageArray,
					vcvt_f16_f32(dstFloat[currentDisparity]));
		}
		else
		{
			storePackedDataUnaligned<float16_t, float16x4_t >(destMessageArrayIndex, dstMessageArray,
					vcvt_f16_f32(dstFloat[currentDisparity]));
		}
		if constexpr (OPTIMIZED_INDEXING_SETTING)
		{
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
		}
		else
		{
			destMessageArrayIndex++;
		}
	}
}

#endif /* KERNELBPSTEREOCPU_NEON_H_ */
