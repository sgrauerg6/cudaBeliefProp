/*
 * KernelBpStereoCPU_ARMTemplateSpFuncts.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_
#define KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_


#include "KernelBpStereoCPU.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

inline
float convertFP16ToFloat(float16_t valToConvert)
{
	//seems like simple cast function works
	return (float)valToConvert;
	 /*float16x8_t float16x8Vector = (float16x8_t) { valToConvert, valToConvert, valToConvert, valToConvert, valToConvert, valToConvert, valToConvert, valToConvert};
	 float32x4_t floatVector = vcvt_high_f32_f16(float16x8Vector);

	 static float p[4];
	 vst1q_f32 (p, floatVector);

	 return p[0];*/
}

inline
float16_t convertFloatToFP16(float valToConvert)
{
	//seems like simple cast function works
	return (float16_t)valToConvert;
	/*float16x4_t floatVector = vcvt_f16_f32(vdupq_n_f32(valToConvert));

		 static float16_t p[8];
		 vst1q_f16 (p, vcombine_f16(floatVector, floatVector));

		 return p[0];*/
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<float16_t>(int xVal, int yVal, int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2,
		float16_t* messageUDeviceCurrentCheckerboard1,
		float16_t* messageDDeviceCurrentCheckerboard1,
		float16_t* messageLDeviceCurrentCheckerboard1,
		float16_t* messageRDeviceCurrentCheckerboard1,
		float16_t* messageUDeviceCurrentCheckerboard2,
		float16_t* messageDDeviceCurrentCheckerboard2,
		float16_t* messageLDeviceCurrentCheckerboard2,
		float16_t* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
		int offsetData)
{
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
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (currentLevelProperties.heightLevel - 1)))
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
				dataMessage[currentDisparity] = convertFP16ToFloat(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = convertFP16ToFloat(messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = convertFP16ToFloat(messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = convertFP16ToFloat(messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = convertFP16ToFloat(messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessage[currentDisparity] = convertFP16ToFloat(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = convertFP16ToFloat(messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = convertFP16ToFloat(messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = convertFP16ToFloat(messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = convertFP16ToFloat(messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
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
			indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				messageUDeviceCurrentCheckerboard1[indexWriteTo] = convertFloatToFP16(currentUMessage[currentDisparity]);
				messageDDeviceCurrentCheckerboard1[indexWriteTo] = convertFloatToFP16(currentDMessage[currentDisparity]);
				messageLDeviceCurrentCheckerboard1[indexWriteTo] = convertFloatToFP16(currentLMessage[currentDisparity]);
				messageRDeviceCurrentCheckerboard1[indexWriteTo] = convertFloatToFP16(currentRMessage[currentDisparity]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				messageUDeviceCurrentCheckerboard2[indexWriteTo] = convertFloatToFP16(currentUMessage[currentDisparity]);
				messageDDeviceCurrentCheckerboard2[indexWriteTo] = convertFloatToFP16(currentDMessage[currentDisparity]);
				messageLDeviceCurrentCheckerboard2[indexWriteTo] = convertFloatToFP16(currentLMessage[currentDisparity]);
				messageRDeviceCurrentCheckerboard2[indexWriteTo] = convertFloatToFP16(currentRMessage[currentDisparity]);
			}
		}
	}
}



template<> inline
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<float16_t>(levelProperties& currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, float16_t* dataCostDeviceStereoCheckerboard1, float16_t* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthLevel;
		int xVal = val % currentLevelProperties.widthLevel;
		int indexVal;
		int xInCheckerboard = xVal / 2;

			if (withinImageBoundsCPU(xInCheckerboard, yVal, currentLevelProperties.widthCheckerboardLevel, currentLevelProperties.heightLevel))
			{
				//make sure that it is possible to check every disparity value
				if ((xVal - (NUM_POSSIBLE_DISPARITY_VALUES-1)) >= 0)
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float currentPixelImage1 = 0.0f;
						float currentPixelImage2 = 0.0f;

						if (withinImageBoundsCPU(xVal, yVal, currentLevelProperties.widthLevel, currentLevelProperties.heightLevel))
						{
							currentPixelImage1 = image1PixelsDevice[yVal * currentLevelProperties.widthLevel
									+ xVal];
							currentPixelImage2 = image2PixelsDevice[yVal * currentLevelProperties.widthLevel
									+ (xVal - currentDisparity)];
						}

						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = convertFloatToFP16((float)(lambda_bp * std::min(((float)abs(currentPixelImage1 - currentPixelImage2)), (float)data_k_bp)));
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = convertFloatToFP16((float)(lambda_bp * std::min(((float)abs(currentPixelImage1 - currentPixelImage2)), (float)data_k_bp)));
						}
					}
				}
				else
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = convertFloatToFP16(getZeroValCPU<float>());
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = convertFloatToFP16(getZeroValCPU<float>());
						}
					}
				}
			}
		//}
	}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<> inline
void KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<float16_t>(int checkerboardPart, levelProperties& currentLevelProperties, levelProperties& prevLevelProperties, float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2, float16_t* dataCostDeviceToWriteTo, int offsetNum)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xVal = val % currentLevelProperties.widthCheckerboardLevel;

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
						prevLevelProperties.widthCheckerboardLevel, prevLevelProperties.heightLevel)) {
					for (int currentDisparity = 0;
							currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
							currentDisparity++) {
						dataCostDeviceToWriteTo[retrieveIndexInDataAndMessageCPU(xVal,
								yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
								currentLevelProperties.heightLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)] =
										convertFloatToFP16((convertFP16ToFloat(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
										xValPrev, (yVal * 2),
										prevLevelProperties.paddedWidthCheckerboardLevel,
										prevLevelProperties.heightLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES,
										offsetNum)])
										+ convertFP16ToFloat(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2),
												prevLevelProperties.paddedWidthCheckerboardLevel,
												prevLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)])
										+ convertFP16ToFloat(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												prevLevelProperties.paddedWidthCheckerboardLevel,
												prevLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)])
										+ convertFP16ToFloat(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												prevLevelProperties.paddedWidthCheckerboardLevel,
												prevLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)])));
					}
				}
			}
		//}
	}
}


template<> inline
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<float16_t>(levelProperties& currentLevelProperties, float16_t* messageUDeviceCurrentCheckerboard1, float16_t* messageDDeviceCurrentCheckerboard1, float16_t* messageLDeviceCurrentCheckerboard1,
		float16_t* messageRDeviceCurrentCheckerboard1, float16_t* messageUDeviceCurrentCheckerboard2, float16_t* messageDDeviceCurrentCheckerboard2,
		float16_t* messageLDeviceCurrentCheckerboard2, float16_t* messageRDeviceCurrentCheckerboard2)
{
	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xValInCheckerboard = val % currentLevelProperties.widthCheckerboardLevel;
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
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
					messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
					messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
					messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
				}

				//retrieve the previous message value at each movement at each pixel
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
					messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
					messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
					messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = convertFloatToFP16(getZeroValCPU<float>());
				}
			}
		//}
	}
}

template<> inline
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU<float16_t>(levelProperties& currentLevelProperties, float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2, float16_t* messageUPrevStereoCheckerboard1, float16_t* messageDPrevStereoCheckerboard1, float16_t* messageLPrevStereoCheckerboard1, float16_t* messageRPrevStereoCheckerboard1, float16_t* messageUPrevStereoCheckerboard2, float16_t* messageDPrevStereoCheckerboard2, float16_t* messageLPrevStereoCheckerboard2, float16_t* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice)
{
	//int widthCheckerboard = getCheckerboardWidthCPU<short>(widthLevel);
	//int paddedWidthCheckerboardCurrentLevel = getPaddedCheckerboardWidth(widthCheckerboard);

	#pragma omp parallel for
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel*currentLevelProperties.heightLevel); val++)
	{
		int yVal = val / currentLevelProperties.widthCheckerboardLevel;
		int xVal = val % currentLevelProperties.widthCheckerboardLevel;

		int xValInCheckerboardPart = xVal;

		//first processing from first part of checkerboard
		{
			//adjustment based on checkerboard; need to add 1 to x for odd-numbered rows for final index mapping into disparity images for checkerboard 1
			int	checkerboardPartAdjustment = (yVal%2);

			if (withinImageBoundsCPU(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentLevelProperties.widthLevel, currentLevelProperties.heightLevel))
			{
				if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) && (xValInCheckerboardPart < (currentLevelProperties.widthCheckerboardLevel - checkerboardPartAdjustment)) && (yVal > 0) && (yVal < (currentLevelProperties.heightLevel - 1)))
				{
					// keep track of "best" disparity for current pixel
					int bestDisparity = 0;
					float best_val = INF_BP;
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float val = convertFP16ToFloat(messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal + 1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
								convertFP16ToFloat(messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal - 1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
										convertFP16ToFloat(messageLPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
												convertFP16ToFloat(messageRPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
														convertFP16ToFloat(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);

						if (val < (best_val)) {
							best_val = val;
							bestDisparity = currentDisparity;
						}
					}

					disparityBetweenImagesDevice[yVal*currentLevelProperties.widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
				}
				else
				{
					disparityBetweenImagesDevice[yVal*currentLevelProperties.widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
				}
			}
		}
		//process from part 2 of checkerboard
		{
			//adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
			int	checkerboardPartAdjustment = ((yVal + 1) % 2);

			if (withinImageBoundsCPU(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentLevelProperties.widthLevel, currentLevelProperties.heightLevel))
			{
				if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) && (xValInCheckerboardPart < (currentLevelProperties.widthCheckerboardLevel - checkerboardPartAdjustment)) && (yVal > 0) && (yVal < (currentLevelProperties.heightLevel - 1)))
				{
					// keep track of "best" disparity for current pixel
					int bestDisparity = 0;
					float best_val = INF_BP;
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float val = convertFP16ToFloat(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal + 1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
								convertFP16ToFloat(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, (yVal - 1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
										convertFP16ToFloat(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
												convertFP16ToFloat(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]) +
														convertFP16ToFloat(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xValInCheckerboardPart, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);

						if (val < (best_val))
						{
							best_val = val;
							bestDisparity = currentDisparity;
						}
					}
					disparityBetweenImagesDevice[yVal*currentLevelProperties.widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
				}
				else
				{
					disparityBetweenImagesDevice[yVal*currentLevelProperties.widthLevel + (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
				}
			}
		}
	}
}

#endif



#endif /* KERNELBPSTEREOCPU_ARMTEMPLATESPFUNCTS_H_ */
