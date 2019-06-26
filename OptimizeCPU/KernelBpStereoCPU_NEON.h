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
		float>(int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1,
		float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1,
		float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2,
		float* messageDDeviceCurrentCheckerboard2,
		float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, float disc_k_bp) {
	int numDataInSIMDVector = 4;
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
			float, float32x4_t>(checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1,
			messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1,
			messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2,
			messageDDeviceCurrentCheckerboard2,
			messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, disc_k_bp, numDataInSIMDVector);
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
		short>(int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2,
		short* messageUDeviceCurrentCheckerboard1,
		short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1,
		short* messageUDeviceCurrentCheckerboard2,
		short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2,
		short* messageRDeviceCurrentCheckerboard2, float disc_k_bp) {
	int numDataInSIMDVector = 4;
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
			short, float16x4_t>(checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1,
			messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1,
			messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2,
			messageDDeviceCurrentCheckerboard2,
			messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, disc_k_bp, numDataInSIMDVector);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataAligned<float,
		float32x4_t>(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, float* inData) {
	return vld1q_f32(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataAligned<short,
		float16x4_t>(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, short* inData) {
	return vld1_f16(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline float32x4_t KernelBpStereoCPU::loadPackedDataUnaligned<float,
		float32x4_t>(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, float* inData) {
	return vld1q_f32(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline float16x4_t KernelBpStereoCPU::loadPackedDataUnaligned<short,
		float16x4_t>(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, short* inData) {
	return vld1_f16(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float,
		float32x4_t>(int indexDataStore, float* locationDataStore,
				float32x4_t dataToStore) {
	return vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<short,
		float16x4_t>(int indexDataStore, short* locationDataStore,
				float16x4_t dataToStore) {
	return vst1_f16(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float,
		float32x4_t>(int indexDataStore, float* locationDataStore,
				float32x4_t dataToStore) {
	return vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<short,
		float16x4_t>(int indexDataStore, short* locationDataStore,
				float16x4_t dataToStore) {
	return vst1_f16(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline float32x4_t KernelBpStereoCPU::createSIMDVectorSameData<
		float32x4_t>(float data) {
	return vdupq_n_f32(data);
}

template<> inline float16x4_t KernelBpStereoCPU::createSIMDVectorSameData<
		float16x4_t>(float data) {
	return vget_high_f16(vdupq_n_f16(convertValToDifferentDataTypeIfNeeded<float, float16_t>(disc_k_bp)));
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoSIMD<float32x4_t>(float32x4_t f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	float32x4_t prev;
	float32x4_t vectorAllOneVal = vdupq_n_f32(1.0f);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = vaddq_f32(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
		 f[currentDisparity] = prev;*/
		f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = vaddq_f32(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
	}
}

template<> inline
void KernelBpStereoCPU::msgStereoSIMD<float, float32x4_t>(int xVal, int yVal, levelProperties& currentLevelProperties, float32x4_t messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		float32x4_t messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		float* dstMessageArray, float32x4_t disc_k_bp, bool dataAligned)
{
	// aggregate and find min
	//T minimum = INF_BP;
	float32x4_t minimum = vdupq_n_f32(INF_BP);
	float32x4_t dst[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
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

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = vminnmq_f32(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = vaddq_f32(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)NUM_POSSIBLE_DISPARITY_VALUES));

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
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
#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
				currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
}

// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoSIMD<short, float16x4_t>(int xVal, int yVal, levelProperties& currentLevelProperties, float16x4_t messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], float16x4_t messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		float16x4_t messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], float16x4_t dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		short* dstMessageArray, float16x4_t disc_k_bp, bool dataAligned)
{
	// aggregate and find min
	//T minimum = INF_BP;
	float32x4_t minimum = vdupq_n_f32(INF_BP);
	float32x4_t dstFloat[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
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

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dstFloat[currentDisparity] = vminnmq_f32(minimum, dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = vaddq_f32(valToNormalize, dstFloat[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)NUM_POSSIBLE_DISPARITY_VALUES));

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dstFloat[currentDisparity] = vsubq_f32(dstFloat[currentDisparity],
				valToNormalize);
		if (dataAligned)
		{
			storePackedDataAligned<short, float16x4_t >(destMessageArrayIndex,
					dstMessageArray,
					vcvt_f16_f32(dstFloat[currentDisparity], 0));
		}
		else
		{
			storePackedDataUnaligned<short, float16x4_t >(destMessageArrayIndex,
					dstMessageArray,
					vcvt_f16_f32(dstFloat[currentDisparity], 0));
		}
#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
				currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
}

#endif /* KERNELBPSTEREOCPU_NEON_H_ */
