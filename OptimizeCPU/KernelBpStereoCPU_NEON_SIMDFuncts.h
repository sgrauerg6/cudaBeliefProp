/*
 * KernelBpStereoCPU_NEON_SIMDFuncts.h
 *
 *  Created on: Feb 21, 2021
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_NEON_SIMD_FUNCTS_H_
#define KERNELBPSTEREOCPU_NEON_SIMD_FUNCTS_H_

//this is only used when processing using an ARM CPU with NEON instructions
#include <arm_neon.h>
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

namespace bp_simd_processing
{
	inline void storePackedDataAligned(const unsigned int indexDataStore, float* locationDataStore,
			const float32x4_t& dataToStore) {
		vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataAligned(const unsigned int indexDataStore, float16_t* locationDataStore,
			const float16x4_t& dataToStore) {
		vst1_f16(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataUnaligned(const unsigned int indexDataStore, float* locationDataStore,
			const float32x4_t& dataToStore) {
		vst1q_f32(&locationDataStore[indexDataStore], dataToStore);
	}

	inline void storePackedDataUnaligned(const unsigned int indexDataStore, float16_t* locationDataStore,
			const float16x4_t& dataToStore) {
		vst1_f16(&locationDataStore[indexDataStore], dataToStore);
	}

	//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	template<unsigned int DISP_VALS>
	void dtStereoSIMD(float32x4_t f[DISP_VALS])
	{
		float32x4_t prev;
		float32x4_t vectorAllOneVal = vdupq_n_f32(1.0f);
		for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
		{
			prev = vaddq_f32(f[currentDisparity-1], vectorAllOneVal);
			//prev = f[currentDisparity-1] + (T)1.0;
			/*if (prev < f[currentDisparity])
			 f[currentDisparity] = prev;*/
			f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
		}

		for (int currentDisparity = (int)DISP_VALS-2; currentDisparity >= 0; currentDisparity--)
		{
			//prev = f[currentDisparity+1] + (T)1.0;
			prev = vaddq_f32(f[currentDisparity+1], vectorAllOneVal);
			//if (prev < f[currentDisparity])
			//	f[currentDisparity] = prev;
			f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
		}
	}

	template<unsigned int DISP_VALS>
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
			float32x4_t messageValsNeighbor1[DISP_VALS], float32x4_t messageValsNeighbor2[DISP_VALS],
			float32x4_t messageValsNeighbor3[DISP_VALS], float32x4_t dataCosts[DISP_VALS],
			float* dstMessageArray, const float32x4_t& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = INF_BP;
		float32x4_t minimum = vdupq_n_f32(bp_consts::INF_BP);
		float32x4_t dst[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
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

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity])
			{
				dst[currentDisparity] = minimum;
			}*/
			dst[currentDisparity] = vminnmq_f32(minimum, dst[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = vaddq_f32(valToNormalize, dst[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_,
					currentLevelProperties.heightLevel_, 0,
					DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dst[currentDisparity] = vsubq_f32(dst[currentDisparity],
					valToNormalize);
			if (dataAligned)
			{
				storePackedDataAligned(destMessageArrayIndex,
						dstMessageArray, dst[currentDisparity]);
			} else
			{
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
	void msgStereoSIMD(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
			float16x4_t messageValsNeighbor1[DISP_VALS], float16x4_t messageValsNeighbor2[DISP_VALS],
			float16x4_t messageValsNeighbor3[DISP_VALS], float16x4_t dataCosts[DISP_VALS],
			float16_t* dstMessageArray, const float16x4_t& disc_k_bp, const bool dataAligned)
	{
		// aggregate and find min
		//T minimum = INF_BP;
		float32x4_t minimum = vdupq_n_f32(bp_consts::INF_BP);
		float32x4_t dstFloat[DISP_VALS];

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
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

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			/*if (minimum < dst[currentDisparity])
			{
				dst[currentDisparity] = minimum;
			}*/
			dstFloat[currentDisparity] = vminnmq_f32(minimum, dstFloat[currentDisparity]);

			//valToNormalize += dst[currentDisparity];
			valToNormalize = vaddq_f32(valToNormalize, dstFloat[currentDisparity]);
		}

		//valToNormalize /= DISP_VALS;
		valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)DISP_VALS));

		unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				DISP_VALS);

		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
		{
			//dst[currentDisparity] -= valToNormalize;
			dstFloat[currentDisparity] = vsubq_f32(dstFloat[currentDisparity], valToNormalize);
			if (dataAligned)
			{
				storePackedDataAligned(destMessageArrayIndex, dstMessageArray,
						vcvt_f16_f32(dstFloat[currentDisparity]));
			}
			else
			{
				storePackedDataUnaligned(destMessageArrayIndex, dstMessageArray,
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
};

#endif /* KERNELBPSTEREOCPU_NEON_SIMD_FUNCTS_H_ */
