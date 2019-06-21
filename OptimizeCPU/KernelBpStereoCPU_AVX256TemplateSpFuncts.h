/*
 * KernelBpStereoCPU_AVX256TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoCPU<__m256>(__m256 f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m256 prev;
	__m256 vectorAllOneVal = _mm256_set1_ps(1.0f);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm256_add_ps(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm256_add_ps(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoCPU<__m256d>(__m256d f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m256d prev;
	__m256d vectorAllOneVal = _mm256_set1_pd(1.0);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm256_add_pd(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
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
void KernelBpStereoCPU::msgStereoCPU<__m128i>(__m128i messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m128i messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m128i dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i dst[NUM_POSSIBLE_DISPARITY_VALUES], __m128i disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m256 minimum = _mm256_set1_ps(INF_BP);
	__m256 dstFloat[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dstFloat[currentDisparity] = _mm256_add_ps((_mm256_cvtph_ps(messageValsNeighbor1[currentDisparity])), (_mm256_cvtph_ps(messageValsNeighbor2[currentDisparity])));
		dstFloat[currentDisparity] = _mm256_add_ps(dstFloat[currentDisparity], _mm256_cvtph_ps(messageValsNeighbor3[currentDisparity]));
		dstFloat[currentDisparity] = _mm256_add_ps(dstFloat[currentDisparity], _mm256_cvtph_ps(dataCosts[currentDisparity]));
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_ps(minimum, dstFloat[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m256>(dstFloat);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_ps(minimum, _mm256_cvtph_ps(disc_k_bp));

	// normalize
	//T valToNormalize = 0;
	__m256 valToNormalize = _mm256_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dstFloat[currentDisparity] = _mm256_min_ps(minimum, dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_ps(valToNormalize, dstFloat[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_cvtps_ph(_mm256_sub_ps(dstFloat[currentDisparity], valToNormalize), 0);
	}
}

// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoCPU<__m256>(__m256 messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m256 messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m256 dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 dst[NUM_POSSIBLE_DISPARITY_VALUES], __m256 disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m256 minimum = _mm256_set1_ps(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm256_add_ps(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm256_add_ps(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm256_add_ps(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_ps(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m256>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_ps(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m256 valToNormalize = _mm256_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm256_min_ps(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_ps(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_sub_ps(dst[currentDisparity], valToNormalize);
	}
}


// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoCPU<__m256d>(__m256d messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m256d messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m256d dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d dst[NUM_POSSIBLE_DISPARITY_VALUES], __m256d disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m256d minimum = _mm256_set1_pd(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm256_add_pd(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm256_add_pd(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm256_add_pd(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_pd(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m256d>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_pd(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m256d valToNormalize = _mm256_set1_pd(0.0);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm256_min_pd(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_pd(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_pd(valToNormalize, _mm256_set1_pd((double)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_sub_pd(dst[currentDisparity], valToNormalize);
	}
}


#endif /* KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_ */
