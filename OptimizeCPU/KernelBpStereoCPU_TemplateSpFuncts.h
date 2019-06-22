#include "KernelBpStereoCPU.h"

#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<short>(
		short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2,
		short* messageUDeviceCurrentCheckerboard1,
		short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1,
		short* messageUDeviceCurrentCheckerboard2,
		short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2,
		short* messageRDeviceCurrentCheckerboard2,
		int widthLevelCheckerboardPart, int heightLevel,
		int checkerboardToUpdate, int xVal, int yVal, int offsetData, float disc_k_bp)
{
	int paddedWidthCheckerboardCurrentLevel = getPaddedCheckerboardWidth(widthLevelCheckerboardPart);

	int indexWriteTo;
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_1)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_2
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	{
		float prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		float dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				dataMessage[currentDisparity] = _cvtsh_ss(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = _cvtsh_ss(messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = _cvtsh_ss(messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = _cvtsh_ss(messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = _cvtsh_ss(messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessage[currentDisparity] = _cvtsh_ss(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = _cvtsh_ss(messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = _cvtsh_ss(messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = _cvtsh_ss(messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = _cvtsh_ss(messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}

		float currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		float currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		//uses the previous message values and data cost to calculate the current message values and store the results
		runBPIterationInOutDataInLocalMemCPU<float>(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							currentUMessage, currentDMessage, currentLMessage, currentRMessage, (float)disc_k_bp);

		//write the calculated message values to global memory
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				messageUDeviceCurrentCheckerboard1[indexWriteTo] = _cvtss_sh(currentUMessage[currentDisparity], 0);
				messageDDeviceCurrentCheckerboard1[indexWriteTo] = _cvtss_sh(currentDMessage[currentDisparity], 0);
				messageLDeviceCurrentCheckerboard1[indexWriteTo] = _cvtss_sh(currentLMessage[currentDisparity], 0);
				messageRDeviceCurrentCheckerboard1[indexWriteTo] = _cvtss_sh(currentRMessage[currentDisparity], 0);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				messageUDeviceCurrentCheckerboard2[indexWriteTo] = _cvtss_sh(currentUMessage[currentDisparity], 0);
				messageDDeviceCurrentCheckerboard2[indexWriteTo] = _cvtss_sh(currentDMessage[currentDisparity], 0);
				messageLDeviceCurrentCheckerboard2[indexWriteTo] = _cvtss_sh(currentLMessage[currentDisparity], 0);
				messageRDeviceCurrentCheckerboard2[indexWriteTo] = _cvtss_sh(currentRMessage[currentDisparity], 0);
			}
		}
	}
}


template<> inline
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<short>(float* image1PixelsDevice, float* image2PixelsDevice, short* dataCostDeviceStereoCheckerboard1, short* dataCostDeviceStereoCheckerboard2, int widthImages, int heightImages, float lambda_bp, float data_k_bp)
{
int imageCheckerboardWidth = getCheckerboardWidthCPU<float>(widthImages);
int paddedWidthCheckerboardCurrentLevel = getPaddedCheckerboardWidth(imageCheckerboardWidth);

	#pragma omp parallel for
	for (int val = 0; val < (widthImages*heightImages); val++)
	{
		int yVal = val / widthImages;
		int xVal = val % widthImages;
			int indexVal;
			int xInCheckerboard = xVal / 2;

			if (withinImageBoundsCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages))
			{
				//make sure that it is possible to check every disparity value
				if ((xVal - (NUM_POSSIBLE_DISPARITY_VALUES-1)) >= 0)
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float currentPixelImage1 = 0.0f;
						float currentPixelImage2 = 0.0f;

						if (withinImageBoundsCPU(xVal, yVal, widthImages, heightImages))
						{
							currentPixelImage1 = image1PixelsDevice[yVal * widthImages
									+ xVal];
							currentPixelImage2 = image2PixelsDevice[yVal * widthImages
									+ (xVal - currentDisparity)];
						}

						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = _cvtss_sh((float)(lambda_bp * std::min(((float)abs(currentPixelImage1 - currentPixelImage2)), (float)data_k_bp)), 0);
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = _cvtss_sh((float)(lambda_bp * std::min(((float)abs(currentPixelImage1 - currentPixelImage2)), (float)data_k_bp)), 0);
						}
					}
				}
				else
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = _cvtss_sh(getZeroValCPU<float>(), 0);
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = _cvtss_sh(getZeroValCPU<float>(), 0);
						}
					}
				}
			}
		//}
	}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<short>(short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* dataCostDeviceToWriteTo, int widthCurrentLevel, int heightCurrentLevel, int widthPrevLevel, int heightPrevLevel, int checkerboardPart, int offsetNum)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<short>(widthCurrentLevel);
	int widthCheckerboardPrevLevel = getCheckerboardWidthCPU<short>(widthPrevLevel);
	int paddedWidthCheckerboardCurrentLevel = getPaddedCheckerboardWidth(widthCheckerboardCurrentLevel);
	int paddedWidthCheckerboardPrevLevel = getPaddedCheckerboardWidth(widthCheckerboardPrevLevel);

	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboardCurrentLevel*heightCurrentLevel); val++)
	{
		int yVal = val / widthCheckerboardCurrentLevel;
		int xVal = val % widthCheckerboardCurrentLevel;

	/*for (int yVal = 0; yVal < heightCurrentLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthCheckerboardCurrentLevel; xVal++)
		{*/
			//if (withinImageBoundsCPU(xVal, yVal, widthCheckerboardCurrentLevel,
			//		heightCurrentLevel))
			{
				//add 1 or 0 to the x-value depending on checkerboard part and row adding to; CHECKERBOARD_PART_1 with slot at (0, 0) has adjustment of 0 in row 0,
				//while CHECKERBOARD_PART_2 with slot at (0, 1) has adjustment of 1 in row 0
				int checkerboardPartAdjustment = 0;

				if (checkerboardPart == CHECKERBOARD_PART_1) {
					checkerboardPartAdjustment = (yVal % 2);
				} else if (checkerboardPart == CHECKERBOARD_PART_2) {
					checkerboardPartAdjustment = ((yVal + 1) % 2);
				}

				//the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
				int xValPrev = xVal * 2 + checkerboardPartAdjustment;

				if (withinImageBoundsCPU(xValPrev, (yVal * 2 + 1),
						widthCheckerboardPrevLevel, heightPrevLevel)) {
					for (int currentDisparity = 0;
							currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
							currentDisparity++) {
						dataCostDeviceToWriteTo[retrieveIndexInDataAndMessageCPU(xVal,
								yVal, paddedWidthCheckerboardCurrentLevel,
								heightCurrentLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)] =
										_cvtss_sh((_cvtsh_ss(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
										xValPrev, (yVal * 2),
										paddedWidthCheckerboardPrevLevel,
										heightPrevLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES,
										offsetNum)])
										+ _cvtsh_ss(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2),
												paddedWidthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)])
										+ _cvtsh_ss(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												paddedWidthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)])
										+ _cvtsh_ss(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												paddedWidthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)])), 0);
					}
				}
			}
		//}
	}
}


template<> inline
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<short>(short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1, short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1, short* messageUDeviceCurrentCheckerboard2, short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2, short* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel)
{
	int paddedWidthCheckerboardCurrentLevel = getPaddedCheckerboardWidth(widthCheckerboardAtLevel);

	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboardAtLevel*heightLevel); val++)
	{
		int yVal = val / widthCheckerboardAtLevel;
		int xValInCheckerboard = val % widthCheckerboardAtLevel;
	/*for (int yVal = 0; yVal < heightLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xValInCheckerboard = 0;
				xValInCheckerboard < widthCheckerboardAtLevel;
				xValInCheckerboard++)
		{*/
			//if (withinImageBoundsCPU(xValInCheckerboard, yVal,
			//		widthCheckerboardAtLevel, heightLevel))
			{
				//initialize message values in both checkerboards

				//set the message value at each pixel for each disparity to 0
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
					messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
					messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
					messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
				}

				//retrieve the previous message value at each movement at each pixel
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
					messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
					messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
					messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, paddedWidthCheckerboardCurrentLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = _cvtss_sh(getZeroValCPU<float>(), 0);
				}
			}
		//}
	}
}


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<float>(float* dataCostStereoCheckerboard1,
		float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1,
		float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1,
		float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2,
		float* messageDDeviceCurrentCheckerboard2,
		float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	//if (widthLevel >= 32)
	{
	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX256<float>(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);
	}
	/*else
	{
		int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

				#pragma omp parallel for
				for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
				{
					int yVal = val / widthCheckerboardCurrentLevel;
					int xVal = val % widthCheckerboardCurrentLevel;
				//for (int yVal = 0; yVal < heightLevel; yVal++)
				//{
				//	#pragma omp parallel for
				//	for (int xVal = 0; xVal < widthLevel / 2; xVal++)
				//	{
						//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
							runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<float>(dataCostStereoCheckerboard1,
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
						//}
					//}
				}
	}*/

#elif CPU_OPTIMIZATION_SETTING == USE_AVX_512

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX512<float>(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
				messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);
#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<float>(dataCostStereoCheckerboard1,
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
				//}
			//}
		}
#endif
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<short>(short* dataCostStereoCheckerboard1,
				short* dataCostStereoCheckerboard2,
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
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	//if (widthLevel >= 32)
		{
	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX256<short>(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);
		}
	/*else
	{
		int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

					#pragma omp parallel for
					for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
					{
						int yVal = val / widthCheckerboardCurrentLevel;
						int xVal = val % widthCheckerboardCurrentLevel;
					//for (int yVal = 0; yVal < heightLevel; yVal++)
					//{
					//	#pragma omp parallel for
					//	for (int xVal = 0; xVal < widthLevel / 2; xVal++)
					//	{
							//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
								runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<short>(dataCostStereoCheckerboard1,
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
							//}
						//}
					}
	}*/

#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

			#pragma omp parallel for
			for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
			{
				int yVal = val / widthCheckerboardCurrentLevel;
				int xVal = val % widthCheckerboardCurrentLevel;
			/*for (int yVal = 0; yVal < heightLevel; yVal++)
			{
				#pragma omp parallel for
				for (int xVal = 0; xVal < widthLevel / 2; xVal++)
				{*/
					//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
						runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<short>(dataCostStereoCheckerboard1,
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
					//}
				//}
			}

#endif
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<double>(double* dataCostStereoCheckerboard1,
		double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1,
		double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1,
		double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2,
		double* messageDDeviceCurrentCheckerboard2,
		double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseAVX256<double>(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<double>(dataCostStereoCheckerboard1,
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
				//}
			//}
		}
#endif
}


template<> inline
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<short>(short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1, short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, short* messageUPrevStereoCheckerboard2, short* messageDPrevStereoCheckerboard2, short* messageLPrevStereoCheckerboard2, short* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
	int widthCheckerboard = getCheckerboardWidthCPU<short>(widthLevel);
	int paddedWidthCheckerboardCurrentLevel = getPaddedCheckerboardWidth(widthCheckerboard);

	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboard*heightLevel); val++)
	{
		int yVal = val / widthCheckerboard;
		int xVal = val % widthCheckerboard;

		int xValInCheckerboardPart = xVal;

		//first processing from first part of checkerboard
		{
			//adjustment based on checkerboard; need to add 1 to x for odd-numbered rows for final index mapping into disparity images for checkerboard 1
			int checkerboardPartAdjustment = (yVal%2);

			if (withinImageBoundsCPU(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, widthLevel, heightLevel))
			{
				if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) && (xValInCheckerboardPart < (widthCheckerboard - checkerboardPartAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
				{
					// keep track of "best" disparity for current pixel
					int bestDisparity = 0;
					float best_val = INF_BP;
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float val = _cvtsh_ss(messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal + 1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal - 1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(messageLPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss( messageRPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);

						if (val < (best_val)) {
							best_val = val;
							bestDisparity = currentDisparity;
						}
					}

					disparityBetweenImagesDevice[yVal*widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
				}
				else
				{
					disparityBetweenImagesDevice[yVal*widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
				}
			}
		}
		//process from part 2 of checkerboard
		{
			//adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
			int checkerboardPartAdjustment = ((yVal + 1) % 2);

			if (withinImageBoundsCPU(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, widthLevel, heightLevel))
			{
				if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) && (xValInCheckerboardPart < (widthCheckerboard - checkerboardPartAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
				{
					// keep track of "best" disparity for current pixel
					int bestDisparity = 0;
					float best_val = INF_BP;
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float val = _cvtsh_ss(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal + 1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal - 1), paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
						_cvtsh_ss(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, yVal, paddedWidthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);

						if (val < (best_val))
						{
							best_val = val;
							bestDisparity = currentDisparity;
						}
					}
					disparityBetweenImagesDevice[yVal*widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
				}
				else
				{
					disparityBetweenImagesDevice[yVal*widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
				}
			}
		}
	}
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
