/*
 * KernelBpStereoCPU_AVX512TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_

#include <x86intrin.h>
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

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
		float* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
{
	int numDataInSIMDVector = 16;
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<
			float, __m512 >(checkerboardToUpdate, currentLevelProperties,
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
		short* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
{
	int numDataInSIMDVector = 16;
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<
			short, __m256i >(checkerboardToUpdate, currentLevelProperties,
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

template<> inline __m512 KernelBpStereoCPU::loadPackedDataAligned<float, __m512 >(
		int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, float* inData) {
	return _mm512_load_ps(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m256i KernelBpStereoCPU::loadPackedDataAligned<short,
		__m256i >(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, short* inData) {
	return _mm256_load_si256(
			(__m256i *) &inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m512 KernelBpStereoCPU::loadPackedDataUnaligned<float,
		__m512 >(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, float* inData) {
	return _mm512_loadu_ps(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m256i KernelBpStereoCPU::loadPackedDataUnaligned<short,
		__m256i >(int x, int y, int currentDisparity,
		levelProperties& currentLevelProperties, short* inData) {
	return _mm256_loadu_si256(
			(__m256i *) &inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float, __m512 >(
		int indexDataStore, float* locationDataStore, __m512 dataToStore) {
	return _mm512_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<short, __m256i >(
		int indexDataStore, short* locationDataStore, __m256i dataToStore) {
	return _mm256_store_si256((__m256i *) &locationDataStore[indexDataStore],
			dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float, __m512 >(
		int indexDataStore, float* locationDataStore, __m512 dataToStore) {
	return _mm512_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<short,
		__m256i >(int indexDataStore, short* locationDataStore,
		__m256i dataToStore) {
	return _mm256_storeu_si256((__m256i *) &locationDataStore[indexDataStore],
			dataToStore);
}

template<> inline __m512 KernelBpStereoCPU::createSIMDVectorSameData<__m512 >(
		float data) {
	return _mm512_set1_ps(data);
}

template<> inline __m256i KernelBpStereoCPU::createSIMDVectorSameData<__m256i >(
		float data) {
	return _mm512_cvtps_ph(_mm512_set1_ps(data), 0);
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoSIMD<__m512 >(
		__m512 f[NUM_POSSIBLE_DISPARITY_VALUES]) {
	__m512 prev;
	__m512 vectorAllOneVal = _mm512_set1_ps(1.0f);
	for (int currentDisparity = 1;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//prev = f[currentDisparity-1] + (T)1.0;
		prev = _mm512_add_ps(f[currentDisparity - 1], vectorAllOneVal);

		/*if (prev < f[currentDisparity])
		 f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= 0; currentDisparity--) {
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm512_add_ps(f[currentDisparity + 1], vectorAllOneVal);

		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoSIMD<__m512d >(
		__m512d f[NUM_POSSIBLE_DISPARITY_VALUES]) {
	__m512d prev;
	__m512d vectorAllOneVal = _mm512_set1_pd(1.0);
	for (int currentDisparity = 1;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//prev = f[currentDisparity-1] + (T)1.0;
		prev = _mm512_add_pd(f[currentDisparity - 1], vectorAllOneVal);

		/*if (prev < f[currentDisparity])
		 f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= 0; currentDisparity--) {
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm512_add_pd(f[currentDisparity + 1], vectorAllOneVal);

		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
	}
}

// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoSIMD<float, __m512 >(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		__m512 messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 dataCosts[NUM_POSSIBLE_DISPARITY_VALUES], float* dstMessageArray,
		__m512 disc_k_bp, bool dataAligned)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512 minimum = _mm512_set1_ps(INF_BP);
	__m512 dst[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = _mm512_add_ps(
				messageValsNeighbor1[currentDisparity],
				messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity],
				messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity],
				dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_ps(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<__m512 >(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_ps(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m512 valToNormalize = _mm512_set1_ps(0.0f);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		/*if (minimum < dst[currentDisparity])
		 {
		 dst[currentDisparity] = minimum;
		 }*/
		dst[currentDisparity] = _mm512_min_ps(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_ps(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_ps(valToNormalize,
			_mm512_set1_ps((float) NUM_POSSIBLE_DISPARITY_VALUES));

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm512_sub_ps(dst[currentDisparity],
				valToNormalize);
		if (dataAligned) {
			storePackedDataAligned<float, __m512 >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		} else {
			storePackedDataUnaligned<float, __m512 >(destMessageArrayIndex,
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
void KernelBpStereoCPU::msgStereoSIMD<double, __m512d >(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		__m512d messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		double* dstMessageArray, __m512d disc_k_bp, bool dataAligned)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512d minimum = _mm512_set1_pd(INF_BP);
	__m512d dst[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = _mm512_add_pd(
				messageValsNeighbor1[currentDisparity],
				messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity],
				messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity],
				dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_pd(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<__m512d >(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_pd(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m512d valToNormalize = _mm512_set1_pd(0.0);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		/*if (minimum < dst[currentDisparity])
		 {
		 dst[currentDisparity] = minimum;
		 }*/
		dst[currentDisparity] = _mm512_min_pd(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_pd(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_pd(valToNormalize,
			_mm512_set1_pd((double) NUM_POSSIBLE_DISPARITY_VALUES));

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm512_sub_pd(dst[currentDisparity],
					valToNormalize);
			if (dataAligned)
			{
				storePackedDataAligned<double, __m512d >(destMessageArrayIndex,
						dstMessageArray,
						dst[currentDisparity]);
			}
			else
			{
				storePackedDataUnaligned<double, __m512d >(destMessageArrayIndex,
						dstMessageArray,
						dst[currentDisparity]);
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
void KernelBpStereoCPU::msgStereoSIMD<short, __m256i >(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		__m256i messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256i messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256i messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256i dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		short* dstMessageArray, __m256i disc_k_bp, bool dataAligned) {
	// aggregate and find min
	//T minimum = INF_BP;
	__m512 minimum = _mm512_set1_ps(INF_BP);
	__m512 dstFloat[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dstFloat[currentDisparity] = _mm512_add_ps(
				(_mm512_cvtph_ps(messageValsNeighbor1[currentDisparity])),
				(_mm512_cvtph_ps(messageValsNeighbor2[currentDisparity])));
		dstFloat[currentDisparity] = _mm512_add_ps(dstFloat[currentDisparity],
				_mm512_cvtph_ps(messageValsNeighbor3[currentDisparity]));
		dstFloat[currentDisparity] = _mm512_add_ps(dstFloat[currentDisparity],
				_mm512_cvtph_ps(dataCosts[currentDisparity]));

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_ps(minimum, dstFloat[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<__m512 >(dstFloat);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_ps(minimum, _mm512_cvtph_ps(disc_k_bp));

	// normalize
	//T valToNormalize = 0;
	__m512 valToNormalize = _mm512_set1_ps(0.0f);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		/*if (minimum < dst[currentDisparity])
		 {
		 dst[currentDisparity] = minimum;
		 }*/
		dstFloat[currentDisparity] = _mm512_min_ps(minimum,
				dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_ps(valToNormalize,
				dstFloat[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_ps(valToNormalize,
			_mm512_set1_ps((float) NUM_POSSIBLE_DISPARITY_VALUES));

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		//dst[currentDisparity] -= valToNormalize;
		dstFloat[currentDisparity] = _mm512_sub_ps(dstFloat[currentDisparity],
				valToNormalize);
		if (dataAligned) {
			storePackedDataAligned<short, __m256i >(destMessageArrayIndex,
					dstMessageArray,
					_mm512_cvtps_ph(dstFloat[currentDisparity], 0));
		} else {
			storePackedDataUnaligned<short, __m256i >(destMessageArrayIndex,
					dstMessageArray,
					_mm512_cvtps_ph(dstFloat[currentDisparity], 0));
		}
#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
				currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
}

#endif /* KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_ */
