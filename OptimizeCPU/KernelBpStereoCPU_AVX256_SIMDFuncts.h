/*
 * KernelBpStereoCPU_AVX256_SIMDFuncts.h
 *
 *  Created on: Feb 13, 2021
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX256_SIMD_FUNCTS_H_
#define KERNELBPSTEREOCPU_AVX256_SIMD_FUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

namespace bp_simd_processing
{
	inline void storePackedDataAligned(
			const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore) {
		_mm256_store_ps(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataAligned(
			const unsigned int indexDataStore, short* locationDataStore, const __m128i& dataToStore) {
		_mm_store_si128((__m128i *) (&locationDataStore[indexDataStore]), dataToStore);
	}

	inline void storePackedDataAligned(
			const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
		_mm256_store_pd(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataUnaligned(
			const unsigned int indexDataStore, float* locationDataStore, const __m256& dataToStore) {
		_mm256_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataUnaligned(
			const unsigned int indexDataStore, short* locationDataStore, const __m128i& dataToStore) {
		_mm_storeu_si128((__m128i *) (&locationDataStore[indexDataStore]), dataToStore);
	}

	inline void storePackedDataUnaligned(
			const unsigned int indexDataStore, double* locationDataStore, const __m256d& dataToStore) {
		_mm256_storeu_pd(&locationDataStore[indexDataStore], dataToStore);
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<unsigned int DISP_VALS>
	void dtStereoSIMD(__m256 f[DISP_VALS])
	{
		__m256 prev;
		const __m256 vectorAllOneVal = _mm256_set1_ps(1.0f);
		for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//prev = f[currentDisparity-1] + (T)1.0;
			prev = _mm256_add_ps(f[currentDisparity - 1], vectorAllOneVal);

			/*if (prev < f[currentDisparity])
						f[currentDisparity] = prev;*/
			f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)DISP_VALS-2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = _mm256_add_ps(f[currentDisparity + 1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
		}
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<unsigned int DISP_VALS>
	void dtStereoSIMD(__m256d f[DISP_VALS])
	{
		__m256d prev;
		const __m256d vectorAllOneVal = _mm256_set1_pd(1.0);
		for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//prev = f[currentDisparity-1] + (T)1.0;
			prev = _mm256_add_pd(f[currentDisparity-1], vectorAllOneVal);

			/*if (prev < f[currentDisparity])
						f[currentDisparity] = prev;*/
			f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)DISP_VALS-2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = _mm256_add_pd(f[currentDisparity+1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
		}
	}

	// compute current message
	template<unsigned int DISP_VALS>
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m128i messageValsNeighbor1[DISP_VALS],
			__m128i messageValsNeighbor2[DISP_VALS],
			__m128i messageValsNeighbor3[DISP_VALS],
			__m128i dataCosts[DISP_VALS],
			short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = bp_consts::INF_BP;
		__m256 minimum = _mm256_set1_ps(bp_consts::INF_BP);
		__m256 dstFloat[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
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
		dtStereoSIMD<DISP_VALS>(dstFloat);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm256_add_ps(minimum, _mm256_cvtph_ps(disc_k_bp));

		// normalize
		//T valToNormalize = 0;
		__m256 valToNormalize = _mm256_set1_ps(0.0f);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity]) {
				dst[currentDisparity] = minimum;
			}*/
			dstFloat[currentDisparity] = _mm256_min_ps(minimum, dstFloat[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm256_add_ps(valToNormalize, dstFloat[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dstFloat[currentDisparity] = _mm256_sub_ps(dstFloat[currentDisparity], valToNormalize);

			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, _mm256_cvtps_ph(dstFloat[currentDisparity], 0));
			}
			else {
				storePackedDataUnaligned(destMessageArrayIndex,
						dstMessageArray, _mm256_cvtps_ph(dstFloat[currentDisparity], 0));
			}

			if constexpr (OPTIMIZED_INDEXING_SETTING) {
				destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
			}
			else {
				destMessageArrayIndex++;
			}
		}
	}

	// compute current message
	template<unsigned int DISP_VALS>
	void msgStereoSIMD(
			const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m256 messageValsNeighbor1[DISP_VALS],
			__m256 messageValsNeighbor2[DISP_VALS],
			__m256 messageValsNeighbor3[DISP_VALS],
			__m256 dataCosts[DISP_VALS],
			float* dstMessageArray, const __m256& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = bp_consts::INF_BP;
		__m256 minimum = _mm256_set1_ps(bp_consts::INF_BP);
		__m256 dst[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
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
		dtStereoSIMD<DISP_VALS>(dst);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm256_add_ps(minimum, disc_k_bp);

		// normalize
		//T valToNormalize = 0;
		__m256 valToNormalize = _mm256_set1_ps(0.0f);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity]) {
				dst[currentDisparity] = minimum;
			}*/
			dst[currentDisparity] = _mm256_min_ps(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm256_add_ps(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, 0,
					DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm256_sub_ps(dst[currentDisparity], valToNormalize);

			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}
			else {
				storePackedDataUnaligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}

			if constexpr (OPTIMIZED_INDEXING_SETTING) {
				destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
			}
			else {
				destMessageArrayIndex++;
			}
		}
	}


	// compute current message
	template<unsigned int DISP_VALS>
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m256d messageValsNeighbor1[DISP_VALS],
			__m256d messageValsNeighbor2[DISP_VALS],
			__m256d messageValsNeighbor3[DISP_VALS],
			__m256d dataCosts[DISP_VALS],
			double* dstMessageArray, const __m256d& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = bp_consts::INF_BP;
		__m256d minimum = _mm256_set1_pd(bp_consts::INF_BP);
		__m256d dst[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
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
		dtStereoSIMD<DISP_VALS>(dst);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm256_add_pd(minimum, disc_k_bp);

		// normalize
		//T valToNormalize = 0;
		__m256d valToNormalize = _mm256_set1_pd(0.0);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity]) {
				dst[currentDisparity] = minimum;
			}*/
			dst[currentDisparity] = _mm256_min_pd(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm256_add_pd(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm256_div_pd(valToNormalize, _mm256_set1_pd((double)DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, 0,
					DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm256_sub_pd(dst[currentDisparity], valToNormalize);

			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}
			else {
				storePackedDataUnaligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}

			if constexpr (OPTIMIZED_INDEXING_SETTING) {
				destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
			}
			else {
				destMessageArrayIndex++;
			}
		}
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	//TODO: look into defining function in .cpp file so don't need to declare inline
	inline void dtStereoSIMD(__m256* f, const unsigned int bpSettingsDispVals)
	{
		__m256 prev;
		const __m256 vectorAllOneVal = _mm256_set1_ps(1.0f);
		for (unsigned int currentDisparity = 1; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//prev = f[currentDisparity-1] + (T)1.0;
			prev = _mm256_add_ps(f[currentDisparity - 1], vectorAllOneVal);

			/*if (prev < f[currentDisparity])
						f[currentDisparity] = prev;*/
			f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)bpSettingsDispVals-2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = _mm256_add_ps(f[currentDisparity + 1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
		}
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	inline void dtStereoSIMD(__m256d* f, const unsigned int bpSettingsDispVals)
	{
		__m256d prev;
		const __m256d vectorAllOneVal = _mm256_set1_pd(1.0);
		for (unsigned int currentDisparity = 1; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//prev = f[currentDisparity-1] + (T)1.0;
			prev = _mm256_add_pd(f[currentDisparity-1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//			f[currentDisparity] = prev;
			f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)bpSettingsDispVals-2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = _mm256_add_pd(f[currentDisparity+1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
		}
	}

	// compute current message
	inline void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m128i* messageValsNeighbor1,
			__m128i* messageValsNeighbor2,
			__m128i* messageValsNeighbor3,
			__m128i* dataCosts,
			short* dstMessageArray, const __m128i& disc_k_bp, const bool dataAligned,
			const unsigned int bpSettingsDispVals)
	{
		// aggregate and find min
		//T minimum = bp_consts::INF_BP;
		__m256 minimum = _mm256_set1_ps(bp_consts::INF_BP);
		__m256* dstFloat = new __m256[bpSettingsDispVals];

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
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
		dtStereoSIMD(dstFloat, bpSettingsDispVals);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm256_add_ps(minimum, _mm256_cvtph_ps(disc_k_bp));

		// normalize
		//T valToNormalize = 0;
		__m256 valToNormalize = _mm256_set1_ps(0.0f);

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//if (minimum < dst[currentDisparity]) {
			//	dst[currentDisparity] = minimum;
			//}
			dstFloat[currentDisparity] = _mm256_min_ps(minimum, dstFloat[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm256_add_ps(valToNormalize, dstFloat[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)bpSettingsDispVals));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				bpSettingsDispVals);

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dstFloat[currentDisparity] = _mm256_sub_ps(dstFloat[currentDisparity], valToNormalize);

			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, _mm256_cvtps_ph(dstFloat[currentDisparity], 0));
			}
			else {
				storePackedDataUnaligned(destMessageArrayIndex,
						dstMessageArray, _mm256_cvtps_ph(dstFloat[currentDisparity], 0));
			}

			if constexpr (OPTIMIZED_INDEXING_SETTING) {
				destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
			}
			else {
				destMessageArrayIndex++;
			}
		}

		delete [] dstFloat;
	}

	// compute current message
	inline void msgStereoSIMD(
			const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m256* messageValsNeighbor1,
			__m256* messageValsNeighbor2,
			__m256* messageValsNeighbor3,
			__m256* dataCosts,
			float* dstMessageArray, const __m256& disc_k_bp, const bool dataAligned,
			const unsigned int bpSettingsDispVals)
	{
		// aggregate and find min
		//T minimum = bp_consts::INF_BP;
		__m256 minimum = _mm256_set1_ps(bp_consts::INF_BP);
		__m256* dst = new __m256[bpSettingsDispVals];

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
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
		dtStereoSIMD(dst, bpSettingsDispVals);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm256_add_ps(minimum, disc_k_bp);

		// normalize
		//T valToNormalize = 0;
		__m256 valToNormalize = _mm256_set1_ps(0.0f);

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//if (minimum < dst[currentDisparity]) {
			//	dst[currentDisparity] = minimum;
			//}
			dst[currentDisparity] = _mm256_min_ps(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm256_add_ps(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)bpSettingsDispVals));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, 0,
					bpSettingsDispVals);

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm256_sub_ps(dst[currentDisparity], valToNormalize);

			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}
			else {
				storePackedDataUnaligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}

			if constexpr (OPTIMIZED_INDEXING_SETTING) {
				destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
			}
			else {
				destMessageArrayIndex++;
			}
		}

		delete [] dst;
	}


	// compute current message
	inline void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m256d* messageValsNeighbor1,
			__m256d* messageValsNeighbor2,
			__m256d* messageValsNeighbor3,
			__m256d* dataCosts,
			double* dstMessageArray, const __m256d& disc_k_bp, const bool dataAligned,
			const unsigned int bpSettingsDispVals)
	{
		// aggregate and find min
		//T minimum = bp_consts::INF_BP;
		__m256d minimum = _mm256_set1_pd(bp_consts::INF_BP);
		__m256d* dst = new __m256d[bpSettingsDispVals];

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
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
		dtStereoSIMD(dst, bpSettingsDispVals);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm256_add_pd(minimum, disc_k_bp);

		// normalize
		//T valToNormalize = 0;
		__m256d valToNormalize = _mm256_set1_pd(0.0);

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//if (minimum < dst[currentDisparity]) {
			//	dst[currentDisparity] = minimum;
			//}
			dst[currentDisparity] = _mm256_min_pd(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm256_add_pd(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm256_div_pd(valToNormalize, _mm256_set1_pd((double)bpSettingsDispVals));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, 0,
					bpSettingsDispVals);

		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm256_sub_pd(dst[currentDisparity], valToNormalize);

			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}
			else {
				storePackedDataUnaligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			}

			if constexpr (OPTIMIZED_INDEXING_SETTING) {
				destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
			}
			else {
				destMessageArrayIndex++;
			}
		}

		delete [] dst;
	}
};

#endif /* KERNELBPSTEREOCPU_AVX256_SIMD_FUNCTS_H_ */
