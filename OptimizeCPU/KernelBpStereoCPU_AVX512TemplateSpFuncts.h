/*
 * KernelBpStereoCPU_AVX512TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoCPU<__m512>(__m512 f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m512 prev;
	__m512 vectorAllOneVal = _mm512_set1_ps(1.0f);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm512_add_ps(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm512_add_ps(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
	}
}


// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoCPU<__m512>(__m512 messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m512 messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m512 dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 dst[NUM_POSSIBLE_DISPARITY_VALUES], __m512 disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512 minimum = _mm512_set1_ps(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm512_add_ps(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_ps(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m512>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_ps(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m512 valToNormalize = _mm512_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm512_min_ps(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_ps(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_ps(valToNormalize, _mm512_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm512_sub_ps(dst[currentDisparity], valToNormalize);
	}
}


//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoCPU<__m512d>(__m512d f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m512d prev;
	__m512d vectorAllOneVal = _mm512_set1_pd(1.0);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm512_add_pd(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm512_add_pd(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
	}
}


// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoCPU<__m512d>(__m512d messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m512d messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m512d dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d dst[NUM_POSSIBLE_DISPARITY_VALUES], __m512d disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512d minimum = _mm512_set1_pd(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm512_add_pd(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_pd(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m512d>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_pd(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m512d valToNormalize = _mm512_set1_pd(0.0);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm512_min_pd(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_pd(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_pd(valToNormalize, _mm512_set1_pd((double)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm512_sub_pd(dst[currentDisparity], valToNormalize);
	}
}


// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoCPU<__m256i>(__m256i messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m256i messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256i messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m256i dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256i dst[NUM_POSSIBLE_DISPARITY_VALUES], __m256i disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512 minimum = _mm512_set1_ps(INF_BP);
	__m512 dstFloat[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dstFloat[currentDisparity] = _mm512_add_ps((_mm512_cvtph_ps(messageValsNeighbor1[currentDisparity])), (_mm512_cvtph_ps(messageValsNeighbor2[currentDisparity])));
		dstFloat[currentDisparity] = _mm512_add_ps(dstFloat[currentDisparity], _mm512_cvtph_ps(messageValsNeighbor3[currentDisparity]));
		dstFloat[currentDisparity] = _mm512_add_ps(dstFloat[currentDisparity], _mm512_cvtph_ps(dataCosts[currentDisparity]));
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_ps(minimum, dstFloat[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m512>(dstFloat);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_ps(minimum, _mm512_cvtph_ps(disc_k_bp));

	// normalize
	//T valToNormalize = 0;
	__m512 valToNormalize = _mm512_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dstFloat[currentDisparity] = _mm512_min_ps(minimum, dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_ps(valToNormalize, dstFloat[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_ps(valToNormalize, _mm512_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm512_cvtps_ph(_mm512_sub_ps(dstFloat[currentDisparity], valToNormalize), 0);
	}
}

//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX512<float>(float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m512 disc_k_bp_vector = _mm512_set1_ps(disc_k_bp);
	int numDataInAvxVector = 16;

	/*#pragma omp parallel for
	for (int val = 0; val < (widthLevel / 2)*heightLevel; val++)
	{
		int yVal = val / (widthLevel / 2);
		int xVal = val % (widthLevel / 2);*/
	//checkerboardAdjustment used for indexing into current checkerboard to update
	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal < endFinal; xVal += numDataInAvxVector)
		{
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xVal > endXAvxStart)
			{
				xVal = endFinal - numDataInAvxVector;
			}

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthCheckerboardCurrentLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			{
				__m512 prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m512 dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm512_loadu_ps(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_ps(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_ps(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_ps(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_ps(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm512_loadu_ps(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_ps(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_ps(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_ps(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_ps(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m512 currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m512>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m512>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m512>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m512>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm512_storeu_ps(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_ps(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_ps(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_ps(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm512_storeu_ps(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_ps(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_ps(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_ps(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			}
			//}
		}

		/*for (int xVal = endXAvxStart + 16; xVal < endFinal; xVal ++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					float>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}*/
	}
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX512<double>(double* dataCostStereoCheckerboard1, double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1, double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2, double* messageDDeviceCurrentCheckerboard2, double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m512d disc_k_bp_vector = _mm512_set1_pd((double)disc_k_bp);
	int numDataInAvxVector = 8;

	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal < endFinal; xVal += numDataInAvxVector)
		{
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xVal > endXAvxStart)
			{
				xVal = endFinal - numDataInAvxVector;
			}

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthCheckerboardCurrentLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			{
				__m512d prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m512d dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm512_loadu_pd(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_pd(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_pd(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_pd(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_pd(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm512_loadu_pd(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_pd(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_pd(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_pd(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_pd(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m512d currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m512d>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m512d>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m512d>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m512d>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm512_storeu_pd(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_pd(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_pd(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_pd(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm512_storeu_pd(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_pd(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_pd(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_pd(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			}
			//}
		}

		/*for (int xVal = endXAvxStart + 8; xVal < endFinal; xVal++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					double>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}*/
	}
}


template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX512<short>(
		short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2,
		short* messageUDeviceCurrentCheckerboard1,
		short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1,
		short* messageUDeviceCurrentCheckerboard2,
		short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2,
		short* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m256i disc_k_bp_vector = _mm512_cvtps_ph(_mm512_set1_ps(disc_k_bp), 0);

	int numDataInAvxVector = 16;
	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal < endFinal; xVal += numDataInAvxVector)
		{
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xVal > endXAvxStart)
			{
				xVal = endFinal - numDataInAvxVector;
			}

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			__m256i dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256i prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256i prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256i prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256i prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] = _mm256_loadu_si256((__m256i*)(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
				}

				__m256i currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256i currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256i currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256i currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m256i>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m256i>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m256i>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m256i>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm256_storeu_si256((__m256i*)&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_si256((__m256i*)&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_si256((__m256i*)&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_si256((__m256i*)&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_storeu_si256((__m256i*)&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_si256((__m256i*)&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_si256((__m256i*)&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_si256((__m256i*)&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}
	}
}

#endif /* KERNELBPSTEREOCPU_AVX512TEMPLATESPFUNCTS_H_ */
