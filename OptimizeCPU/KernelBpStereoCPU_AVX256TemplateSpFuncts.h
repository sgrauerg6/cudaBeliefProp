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

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
		float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard0, float* dataCostStereoCheckerboard1,
		float* messageUDeviceCurrentCheckerboard0, float* messageDDeviceCurrentCheckerboard0,
		float* messageLDeviceCurrentCheckerboard0, float* messageRDeviceCurrentCheckerboard0,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{8u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
		short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		short* dataCostStereoCheckerboard0, short* dataCostStereoCheckerboard1,
		short* messageUDeviceCurrentCheckerboard0, short* messageDDeviceCurrentCheckerboard0,
		short* messageLDeviceCurrentCheckerboard0, short* messageRDeviceCurrentCheckerboard0,
		short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1, short* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{8u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<
	double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		double* dataCostStereoCheckerboard0, double* dataCostStereoCheckerboard1,
		double* messageUDeviceCurrentCheckerboard0, double* messageDDeviceCurrentCheckerboard0,
		double* messageLDeviceCurrentCheckerboard0, double* messageRDeviceCurrentCheckerboard0,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp)
{
	constexpr unsigned int numDataInSIMDVector{4u};
	runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess<double, __m256d, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
			checkerboardToUpdate, currentLevelProperties,
			dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
			messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
			messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
			messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			disc_k_bp, numDataInSIMDVector);
}

template<> inline __m256d KernelBpStereoCPU::loadPackedDataAligned<double, __m256d, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, double* inData) {
	return _mm256_load_pd(
		&inData[retrieveIndexInDataAndMessage(x, y,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, currentDisparity,
			bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m256 KernelBpStereoCPU::loadPackedDataAligned<float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, float* inData) {
	return _mm256_load_ps(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m128i KernelBpStereoCPU::loadPackedDataAligned<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, short* inData) {
	return _mm_load_si128(
			(__m128i *) &inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m256 KernelBpStereoCPU::loadPackedDataUnaligned<float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, float* inData) {
	return _mm256_loadu_ps(
			&inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m128i KernelBpStereoCPU::loadPackedDataUnaligned<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, short* inData) {
	return _mm_loadu_si128(
			(__m128i *) &inData[retrieveIndexInDataAndMessage(x, y,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline __m256d KernelBpStereoCPU::loadPackedDataUnaligned<double, __m256d, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int x, const unsigned int y, const unsigned int currentDisparity,
		const levelProperties& currentLevelProperties, double* inData) {
	return _mm256_loadu_pd(
		&inData[retrieveIndexInDataAndMessage(x, y,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, currentDisparity,
			bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<float, __m256>(
		const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore) {
	_mm256_store_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<short, __m128i>(
		const unsigned int indexDataStore, short* locationDataStore, const __m128i& dataToStore) {
	_mm_store_si128((__m128i *) (&locationDataStore[indexDataStore]), dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataAligned<double, __m256d>(
		const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
	_mm256_store_pd(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<float, __m256>(
		const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore) {
	_mm256_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<short, __m128i>(
		const unsigned int indexDataStore, short* locationDataStore, const __m128i& dataToStore) {
	_mm_storeu_si128((__m128i *) (&locationDataStore[indexDataStore]), dataToStore);
}

template<> inline void KernelBpStereoCPU::storePackedDataUnaligned<double, __m256d>(
		const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
	_mm256_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
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

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoSIMD<__m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(__m256 f[bp_params::NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m256 prev;
	const __m256 vectorAllOneVal = _mm256_set1_ps(1.0f);
	for (unsigned int currentDisparity = 1; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//prev = f[currentDisparity-1] + (T)1.0;
		prev = _mm256_add_ps(f[currentDisparity - 1], vectorAllOneVal);

		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
	}

	for (int currentDisparity = (int)bp_params::NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm256_add_ps(f[currentDisparity + 1], vectorAllOneVal);

		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoSIMD<__m256d, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(__m256d f[bp_params::NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m256d prev;
	const __m256d vectorAllOneVal = _mm256_set1_pd(1.0);
	for (unsigned int currentDisparity = 1; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//prev = f[currentDisparity-1] + (T)1.0;
		prev = _mm256_add_pd(f[currentDisparity-1], vectorAllOneVal);

		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
	}

	for (int currentDisparity = (int)bp_params::NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm256_add_pd(f[currentDisparity+1], vectorAllOneVal);

		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
	}
}

// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoSIMD<short, __m128i, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m128i messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	//T minimum = bp_consts::INF_BP;
	__m256 minimum = _mm256_set1_ps(bp_consts::INF_BP);
	__m256 dstFloat[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dstFloat[currentDisparity] = _mm256_add_ps((_mm256_cvtph_ps(messageValsNeighbor1[currentDisparity])), (_mm256_cvtph_ps(messageValsNeighbor2[currentDisparity])));
		dstFloat[currentDisparity] = _mm256_add_ps(dstFloat[currentDisparity], _mm256_cvtph_ps(messageValsNeighbor3[currentDisparity]));
		dstFloat[currentDisparity] = _mm256_add_ps(dstFloat[currentDisparity], _mm256_cvtph_ps(dataCosts[currentDisparity]));

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_ps(minimum, dstFloat[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<__m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(dstFloat);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_ps(minimum, _mm256_cvtph_ps(disc_k_bp));

	// normalize
	//T valToNormalize = 0;
	__m256 valToNormalize = _mm256_set1_ps(0.0f);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dstFloat[currentDisparity] = _mm256_min_ps(minimum, dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_ps(valToNormalize, dstFloat[currentDisparity]);
	}

	//valToNormalize /= bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)bp_params::NUM_POSSIBLE_DISPARITY_VALUES));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dstFloat[currentDisparity] = _mm256_sub_ps(dstFloat[currentDisparity], valToNormalize);

		if (dataAligned) {
			storePackedDataAligned<short, __m128i>(destMessageArrayIndex,
					dstMessageArray, _mm256_cvtps_ph(dstFloat[currentDisparity], 0));
		}
		else {
			storePackedDataUnaligned<short, __m128i>(destMessageArrayIndex,
					dstMessageArray, _mm256_cvtps_ph(dstFloat[currentDisparity], 0));
		}

		if constexpr (OPTIMIZED_INDEXING_SETTING) {
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel;
		}
		else {
			destMessageArrayIndex++;
		}
	}
}

// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoSIMD<float, __m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m256 messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		float* dstMessageArray, const __m256& disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	//T minimum = bp_consts::INF_BP;
	__m256 minimum = _mm256_set1_ps(bp_consts::INF_BP);
	__m256 dst[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = _mm256_add_ps(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm256_add_ps(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm256_add_ps(dst[currentDisparity], dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_ps(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<__m256, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_ps(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m256 valToNormalize = _mm256_set1_ps(0.0f);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm256_min_ps(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_ps(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)bp_params::NUM_POSSIBLE_DISPARITY_VALUES));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_sub_ps(dst[currentDisparity], valToNormalize);

		if (dataAligned) {
			storePackedDataAligned<float, __m256 >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}
		else {
			storePackedDataUnaligned<float, __m256 >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}

		if constexpr (OPTIMIZED_INDEXING_SETTING) {
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel;
		}
		else {
			destMessageArrayIndex++;
		}
	}
}


// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoSIMD<double, __m256d, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		__m256d messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
		double* dstMessageArray, const __m256d& disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	//T minimum = bp_consts::INF_BP;
	__m256d minimum = _mm256_set1_pd(bp_consts::INF_BP);
	__m256d dst[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = _mm256_add_pd(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm256_add_pd(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm256_add_pd(dst[currentDisparity], dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_pd(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<__m256d, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_pd(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m256d valToNormalize = _mm256_set1_pd(0.0);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm256_min_pd(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_pd(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_pd(valToNormalize, _mm256_set1_pd((double)bp_params::NUM_POSSIBLE_DISPARITY_VALUES));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	for (unsigned int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_sub_pd(dst[currentDisparity], valToNormalize);

		if (dataAligned) {
			storePackedDataAligned<double, __m256d >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}
		else {
			storePackedDataUnaligned<double, __m256d >(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}

		if constexpr (OPTIMIZED_INDEXING_SETTING) {
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel;
		}
		else {
			destMessageArrayIndex++;
		}
	}
}

#endif /* KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_ */
