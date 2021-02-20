/*
 * KernelBpStereoCPU_AVX512_SIMDFuncts.h
 *
 *  Created on: Feb 20, 2021
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX512_SIMD_FUNCTS_H_
#define KERNELBPSTEREOCPU_AVX512_SIMD_FUNCTS_H_
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

namespace bp_simd_processing
{
	inline void storePackedDataAligned(const unsigned int indexDataStore, float* locationDataStore, const __m512& dataToStore) {
		return _mm512_store_ps(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataAligned(const unsigned int indexDataStore, short* locationDataStore, const __m256i& dataToStore) {
		return _mm256_store_si256((__m256i *) &locationDataStore[indexDataStore],
				dataToStore);
	}

	inline void storePackedDataUnaligned(const unsigned int indexDataStore, float* locationDataStore, const __m512& dataToStore) {
		return _mm512_storeu_ps(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataUnaligned(const unsigned int indexDataStore, short* locationDataStore, const __m256i& dataToStore) {
		return _mm256_storeu_si256((__m256i *) &locationDataStore[indexDataStore],
			dataToStore);
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<unsigned int DISP_VALS>
	void dtStereoSIMD(__m512 f[DISP_VALS])
	{
		__m512 prev;
		__m512 vectorAllOneVal = _mm512_set1_ps(1.0f);
		for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//prev = f[currentDisparity-1] + (T)1.0;
			prev = _mm512_add_ps(f[currentDisparity - 1], vectorAllOneVal);

			/*if (prev < f[currentDisparity])
			 f[currentDisparity] = prev;*/
			f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)DISP_VALS - 2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = _mm512_add_ps(f[currentDisparity + 1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
		}
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<unsigned int DISP_VALS>
	void dtStereoSIMD(__m512d f[DISP_VALS])
	{
		__m512d prev;
		__m512d vectorAllOneVal = _mm512_set1_pd(1.0);
		for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//prev = f[currentDisparity-1] + (T)1.0;
			prev = _mm512_add_pd(f[currentDisparity - 1], vectorAllOneVal);

			/*if (prev < f[currentDisparity])
			 f[currentDisparity] = prev;*/
			f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)DISP_VALS - 2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = _mm512_add_pd(f[currentDisparity + 1], vectorAllOneVal);

			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
		}
	}

	// compute current message
	template<unsigned int DISP_VALS>
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties, __m512 messageValsNeighbor1[DISP_VALS],
			__m512 messageValsNeighbor2[DISP_VALS], __m512 messageValsNeighbor3[DISP_VALS],
			__m512 dataCosts[DISP_VALS], float* dstMessageArray,
			const __m512& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = INF_BP;
		__m512 minimum = _mm512_set1_ps(bp_consts::INF_BP);
		__m512 dst[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
			dst[currentDisparity] = _mm512_add_ps(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
			dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
			dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity], dataCosts[currentDisparity]);

			//if (dst[currentDisparity] < minimum)
			//	minimum = dst[currentDisparity];
			minimum = _mm512_min_ps(minimum, dst[currentDisparity]);
		}

		//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
		dtStereoSIMD(dst);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm512_add_ps(minimum, disc_k_bp);

		// normalize
		//T valToNormalize = 0;
		__m512 valToNormalize = _mm512_set1_ps(0.0f);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity]) {
			 dst[currentDisparity] = minimum;
			 }*/
			dst[currentDisparity] = _mm512_min_ps(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm512_add_ps(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm512_div_ps(valToNormalize,
				_mm512_set1_ps((float)DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm512_sub_ps(dst[currentDisparity], valToNormalize);
			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			} else {
				storePackedDataUnaligned(destMessageArrayIndex,
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
	template<unsigned int DISP_VALS>
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
			const levelProperties& currentLevelProperties,
			__m512d messageValsNeighbor1[DISP_VALS],
			__m512d messageValsNeighbor2[DISP_VALS],
			__m512d messageValsNeighbor3[DISP_VALS],
			__m512d dataCosts[DISP_VALS],
			double* dstMessageArray, const __m512d& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = INF_BP;
		__m512d minimum = _mm512_set1_pd(bp_consts::INF_BP);
		__m512d dst[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
			dst[currentDisparity] = _mm512_add_pd(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
			dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
			dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity], dataCosts[currentDisparity]);

			//if (dst[currentDisparity] < minimum)
			//	minimum = dst[currentDisparity];
			minimum = _mm512_min_pd(minimum, dst[currentDisparity]);
		}

		//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
		dtStereoSIMD(dst);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm512_add_pd(minimum, disc_k_bp);

		// normalize
		//T valToNormalize = 0;
		__m512d valToNormalize = _mm512_set1_pd(0.0);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity]) {
			 dst[currentDisparity] = minimum;
			 }*/
			dst[currentDisparity] = _mm512_min_pd(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm512_add_pd(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm512_div_pd(valToNormalize,
				_mm512_set1_pd((double) DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, 0,
					DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = _mm512_sub_pd(dst[currentDisparity], valToNormalize);
			if (dataAligned)
			{
				storePackedDataAligned(destMessageArrayIndex, dstMessageArray, dst[currentDisparity]);
			}
			else
			{
				storePackedDataUnaligned(destMessageArrayIndex, dstMessageArray, dst[currentDisparity]);
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
	template<unsigned int DISP_VALS>
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
			__m256i messageValsNeighbor1[DISP_VALS], __m256i messageValsNeighbor2[DISP_VALS],
			__m256i messageValsNeighbor3[DISP_VALS], __m256i dataCosts[DISP_VALS],
			short* dstMessageArray, const __m256i& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = INF_BP;
		__m512 minimum = _mm512_set1_ps(bp_consts::INF_BP);
		__m512 dstFloat[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
			dstFloat[currentDisparity] = _mm512_add_ps((_mm512_cvtph_ps(messageValsNeighbor1[currentDisparity])),
					(_mm512_cvtph_ps(messageValsNeighbor2[currentDisparity])));
			dstFloat[currentDisparity] = _mm512_add_ps(dstFloat[currentDisparity], _mm512_cvtph_ps(messageValsNeighbor3[currentDisparity]));
			dstFloat[currentDisparity] = _mm512_add_ps(dstFloat[currentDisparity], _mm512_cvtph_ps(dataCosts[currentDisparity]));

			//if (dst[currentDisparity] < minimum)
			//	minimum = dst[currentDisparity];
			minimum = _mm512_min_ps(minimum, dstFloat[currentDisparity]);
		}

		//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
		dtStereoSIMD(dstFloat);

		// truncate
		//minimum += disc_k_bp;
		minimum = _mm512_add_ps(minimum, _mm512_cvtph_ps(disc_k_bp));

		// normalize
		//T valToNormalize = 0;
		__m512 valToNormalize = _mm512_set1_ps(0.0f);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity]) {
			 dst[currentDisparity] = minimum;
			 }*/
			dstFloat[currentDisparity] = _mm512_min_ps(minimum, dstFloat[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = _mm512_add_ps(valToNormalize, dstFloat[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = _mm512_div_ps(valToNormalize,
				_mm512_set1_ps((float) DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dstFloat[currentDisparity] = _mm512_sub_ps(dstFloat[currentDisparity], valToNormalize);
			if (dataAligned) {
				storePackedDataAligned(destMessageArrayIndex, dstMessageArray,
						_mm512_cvtps_ph(dstFloat[currentDisparity], 0));
			} else {
				storePackedDataUnaligned(destMessageArrayIndex, dstMessageArray,
						_mm512_cvtps_ph(dstFloat[currentDisparity], 0));
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
};

#endif /* KERNELBPSTEREOCPU_AVX512_SIMD_FUNCTS_H_ */
