/*
 * KernelBpStereoCPU_AVX256TemplateSpFuncts.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_

#include <x86intrin.h>

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

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX256<float>(int checkerboardToUpdate, levelProperties& currentLevelProperties,
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
	int numDataInAvxVector = 8;
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;
	__m256 disc_k_bp_vector = _mm256_set1_ps(disc_k_bp);

	#pragma omp parallel for
	for (int yVal = 1; yVal < currentLevelProperties.heightLevel - 1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}

		int startX = (checkerboardAdjustment == 1) ? 0 : 1;
		int endFinal = std::min(
				currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment,
				widthCheckerboardRunProcessing);
		int endXAvxStart = (endFinal / numDataInAvxVector) * numDataInAvxVector
				- numDataInAvxVector;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			int xValProcess = xVal;

			//need this check first for case where endXAvxStart is 0 and startX is 1
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xValProcess > endXAvxStart) {
				xValProcess = endFinal - numDataInAvxVector;
			}

			//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
			xValProcess = std::max(startX, xValProcess);

			//check if the memory is aligned for AVX instructions at xValProcess location
			bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(
					xValProcess, numDataInAvxVector);

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			__m256 dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			//load using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								_mm256_load_ps(
										&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm256_load_ps(
										&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm256_load_ps(
										&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								_mm256_load_ps(
										&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm256_load_ps(
										&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm256_load_ps(
										&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								_mm256_loadu_ps(
										&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								_mm256_loadu_ps(
										&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm256_loadu_ps(
										&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}
			}

			__m256 currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m256 currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			msgStereoCPU<__m256 >(prevUMessage, prevLMessage, prevRMessage,
					dataMessage, currentUMessage, disc_k_bp_vector);

			msgStereoCPU<__m256 >(prevDMessage, prevLMessage, prevRMessage,
					dataMessage, currentDMessage, disc_k_bp_vector);

			msgStereoCPU<__m256 >(prevUMessage, prevDMessage, prevRMessage,
					dataMessage, currentRMessage, disc_k_bp_vector);

			msgStereoCPU<__m256 >(prevUMessage, prevDMessage, prevLMessage,
					dataMessage, currentLMessage, disc_k_bp_vector);

			//save using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				//write the calculated message values to global memory
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						_mm256_store_ps(
								&messageUDeviceCurrentCheckerboard1[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm256_store_ps(
								&messageDDeviceCurrentCheckerboard1[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm256_store_ps(
								&messageLDeviceCurrentCheckerboard1[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm256_store_ps(
								&messageRDeviceCurrentCheckerboard1[indexWriteTo],
								currentRMessage[currentDisparity]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_store_ps(
								&messageUDeviceCurrentCheckerboard2[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm256_store_ps(
								&messageDDeviceCurrentCheckerboard2[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm256_store_ps(
								&messageLDeviceCurrentCheckerboard2[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm256_store_ps(
								&messageRDeviceCurrentCheckerboard2[indexWriteTo],
								currentRMessage[currentDisparity]);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						_mm256_storeu_ps(
								&messageUDeviceCurrentCheckerboard1[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm256_storeu_ps(
								&messageDDeviceCurrentCheckerboard1[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm256_storeu_ps(
								&messageLDeviceCurrentCheckerboard1[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm256_storeu_ps(
								&messageRDeviceCurrentCheckerboard1[indexWriteTo],
								currentRMessage[currentDisparity]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_storeu_ps(
								&messageUDeviceCurrentCheckerboard2[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm256_storeu_ps(
								&messageDDeviceCurrentCheckerboard2[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm256_storeu_ps(
								&messageLDeviceCurrentCheckerboard2[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm256_storeu_ps(
								&messageRDeviceCurrentCheckerboard2[indexWriteTo],
								currentRMessage[currentDisparity]);
					}
				}
			}
		}
	}
}


template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX256<short>(int checkerboardToUpdate, levelProperties& currentLevelProperties,
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
	int numDataInAvxVector = 8;
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;
	__m128i disc_k_bp_vector = _mm256_cvtps_ph(_mm256_set1_ps(disc_k_bp), 0);

	#pragma omp parallel for
	for (int yVal = 1; yVal < currentLevelProperties.heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}

		int startX = (checkerboardAdjustment == 1) ? 0 : 1;
		int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment, widthCheckerboardRunProcessing);
		int endXAvxStart = (endFinal / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			int xValProcess = xVal;

			//need this check first for case where endXAvxStart is 0 and startX is 1
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xValProcess > endXAvxStart) {
				xValProcess = endFinal - numDataInAvxVector;
			}

			//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
			xValProcess = std::max(startX, xValProcess);

			//check if the memory is aligned for AVX instructions at xValProcess location
			bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(
					xValProcess, numDataInAvxVector);

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			__m128i dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			//load using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					//load using aligned instructions when possible
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								_mm_load_si128(
										(__m128i *) &dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm_load_si128(
										(__m128i *) &messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm_load_si128(
										(__m128i *) &messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								_mm_load_si128(
										(__m128i *) &dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm_load_si128(
										(__m128i *) &messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm_load_si128(
										(__m128i *) &messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {

					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								_mm_loadu_si128(
										(__m128i *) &messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}
			}

			__m128i currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			msgStereoCPU<__m128i >(prevUMessage, prevLMessage, prevRMessage,
					dataMessage, currentUMessage, disc_k_bp_vector);

			msgStereoCPU<__m128i >(prevDMessage, prevLMessage, prevRMessage,
					dataMessage, currentDMessage, disc_k_bp_vector);

			msgStereoCPU<__m128i >(prevUMessage, prevDMessage, prevRMessage,
					dataMessage, currentRMessage, disc_k_bp_vector);

			msgStereoCPU<__m128i >(prevUMessage, prevDMessage, prevLMessage,
					dataMessage, currentLMessage, disc_k_bp_vector);

			//save using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				//write the calculated message values to global memory
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						_mm_store_si128(
								(__m128i *) &messageUDeviceCurrentCheckerboard1[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm_store_si128(
								(__m128i *) &messageDDeviceCurrentCheckerboard1[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm_store_si128(
								(__m128i *) &messageLDeviceCurrentCheckerboard1[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm_store_si128(
								(__m128i *) &messageRDeviceCurrentCheckerboard1[indexWriteTo],
								currentRMessage[currentDisparity]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm_store_si128(
								(__m128i *) &messageUDeviceCurrentCheckerboard2[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm_store_si128(
								(__m128i *) &messageDDeviceCurrentCheckerboard2[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm_store_si128(
								(__m128i *) &messageLDeviceCurrentCheckerboard2[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm_store_si128(
								(__m128i *) &messageRDeviceCurrentCheckerboard2[indexWriteTo],
								currentRMessage[currentDisparity]);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						_mm_storeu_si128(
								(__m128i *) &messageUDeviceCurrentCheckerboard1[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm_storeu_si128(
								(__m128i *) &messageDDeviceCurrentCheckerboard1[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm_storeu_si128(
								(__m128i *) &messageLDeviceCurrentCheckerboard1[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm_storeu_si128(
								(__m128i *) &messageRDeviceCurrentCheckerboard1[indexWriteTo],
								currentRMessage[currentDisparity]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm_storeu_si128(
								(__m128i *) &messageUDeviceCurrentCheckerboard2[indexWriteTo],
								currentUMessage[currentDisparity]);
						_mm_storeu_si128(
								(__m128i *) &messageDDeviceCurrentCheckerboard2[indexWriteTo],
								currentDMessage[currentDisparity]);
						_mm_storeu_si128(
								(__m128i *) &messageLDeviceCurrentCheckerboard2[indexWriteTo],
								currentLMessage[currentDisparity]);
						_mm_storeu_si128(
								(__m128i *) &messageRDeviceCurrentCheckerboard2[indexWriteTo],
								currentRMessage[currentDisparity]);
					}
				}
			}
		}
	}
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX256<double>(int checkerboardToUpdate, levelProperties& currentLevelProperties,
		double* dataCostStereoCheckerboard1, double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1,
		double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1,
		double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2,
		double* messageDDeviceCurrentCheckerboard2,
		double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
{
	int numDataInAvxVector = 4;
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;
	__m256d disc_k_bp_vector = _mm256_set1_pd((double)disc_k_bp);

	#pragma omp parallel for
	for (int yVal = 1; yVal < currentLevelProperties.heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}

		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endFinal = widthCheckerboardRunProcessing - checkerboardAdjustment;
		int endXAvxStart = (endFinal / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			int xValProcess = xVal;

			//need this check first for case where endXAvxStart is 0 and startX is 1
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xValProcess > endXAvxStart)
			{
				xValProcess = endFinal - numDataInAvxVector;
			}

			//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
			xValProcess = std::max(startX, xValProcess);

			int indexWriteTo;

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
				__m256d prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m256d dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardToUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm256_loadu_pd(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValProcess, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_pd(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xValProcess, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_pd(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xValProcess, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_pd(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xValProcess + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_pd(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xValProcess - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm256_loadu_pd(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValProcess, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_pd(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xValProcess, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_pd(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xValProcess, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_pd(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xValProcess + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_pd(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xValProcess - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m256d currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m256d>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m256d>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m256d>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m256d>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					indexWriteTo = retrieveIndexInDataAndMessageCPU(xValProcess, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1)
					{
						_mm256_storeu_pd(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_pd(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_pd(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_pd(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_storeu_pd(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_pd(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_pd(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_pd(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}
	}
}

#endif /* KERNELBPSTEREOCPU_AVX256TEMPLATESPFUNCTS_H_ */
