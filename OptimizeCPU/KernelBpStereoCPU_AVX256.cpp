#include "KernelBpStereoCPU.h"
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"

void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX256(float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m256 disc_k_bp_vector = _mm256_set1_ps(disc_k_bp);

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
				__m256 prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m256 dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm256_loadu_ps(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_ps(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_ps(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_ps(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_ps(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm256_loadu_ps(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_ps(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_ps(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_ps(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_ps(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m256 currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m256>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m256>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m256>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m256>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm256_storeu_ps(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_ps(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_ps(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_ps(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_storeu_ps(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_ps(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_ps(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_ps(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}

		/*for (int xVal = endXAvxStart + 8; xVal < endFinal; xVal ++)
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


void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUDoubleUseAVX256(double* dataCostStereoCheckerboard1, double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1, double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2, double* messageDDeviceCurrentCheckerboard2, double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m256d disc_k_bp_vector = _mm256_set1_pd((double)disc_k_bp);

	int numDataInAvxVector = 4;
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
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm256_loadu_pd(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_pd(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_pd(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_pd(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_pd(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm256_loadu_pd(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_pd(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_pd(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_pd(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_pd(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
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
					indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
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

		/*for (int xVal = endXAvxStart + 4; xVal < endFinal; xVal ++)
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


void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUShortUseAVX256(
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
	__m128i disc_k_bp_vector = _mm256_cvtps_ph(_mm256_set1_ps(disc_k_bp), 0);

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
			__m128i dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
				}

				__m128i currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m128i currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m128i currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m128i currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m128i>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m128i>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m128i>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m128i>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm_storeu_si128((__m128i*)&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm_storeu_si128((__m128i*)&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}
	}
}


void KernelBpStereoCPU::convertShortToFloatAVX256(float* destinationFloat, short* inputShort, int widthArray, int heightArray)
{
	int numDataInAvxVector = 8;
	#pragma omp parallel for
	for (int yVal = 0; yVal < heightArray; yVal++)
	{
		int startX = 0;
		int endXAvxStart = (((widthArray - startX) / numDataInAvxVector)
				* numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthArray;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			if (xVal > endXAvxStart) {
				xVal = endFinal - numDataInAvxVector;
			}
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++)
			{
				//load __m128i vector and convert to __m256 (set of 8 32-bit floats)
				__m256 data32Bit =
						_mm256_cvtph_ps(
								_mm_loadu_si128((__m128i*)(&inputShort[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthArray, heightArray, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)])));

				//store the __m256
				_mm256_storeu_ps(
						(&destinationFloat[retrieveIndexInDataAndMessageCPU(
								xVal, yVal, widthArray,
								heightArray, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)]), data32Bit);
			}
		}
	}
}

void KernelBpStereoCPU::convertFloatToShortAVX256(short* destinationShort, float* inputFloat, int widthArray, int heightArray)
{
	int numDataInAvxVector = 8;
	#pragma omp parallel for
	for (int yVal = 0; yVal < heightArray; yVal++)
	{
		int startX = 0;
		int endXAvxStart = (((widthArray - startX) / numDataInAvxVector)
				* numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthArray;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			if (xVal > endXAvxStart) {
				xVal = endFinal - numDataInAvxVector;
			}
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++)
			{
				//load __m256 vector and convert to __m128i (that is storing 16-bit floats)
				__m128i data16Bit =
						_mm256_cvtps_ph(
								_mm256_loadu_ps(&inputFloat[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthArray, heightArray, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]), 0);

				//store the 16-bit floats
				_mm_storeu_si128(
						((__m128i*)&destinationShort[retrieveIndexInDataAndMessageCPU(
								xVal, yVal, widthArray,
								heightArray, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)]), data16Bit);
			}
		}
	}
}
