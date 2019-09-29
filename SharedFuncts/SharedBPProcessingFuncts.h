/*
 * SharedBPProcessingFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDBPPROCESSINGFUNCTS_H_
#define SHAREDBPPROCESSINGFUNCTS_H_

#include "SharedUtilFuncts.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../ParameterFiles/bpRunSettings.h"

//typename T is input type, typename U is output type
template<typename T, typename U>
ARCHITECTURE_ADDITION inline U convertValToDifferentDataTypeIfNeeded(T data)
{
	//by default assume same data type and just return data
	return data;
}

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
ARCHITECTURE_ADDITION inline int retrieveIndexInDataAndMessage(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData = 0)
{
#ifdef _WIN32
	//assuming that width includes padding
	if constexpr (OPTIMIZED_INDEXING_SETTING)
#else
	if (OPTIMIZED_INDEXING_SETTING)
#endif
	{
		//indexing is performed in such a way so that the memory accesses as coalesced as much as possible
		return (yVal * width * totalNumDispVals + width * currentDisparity + xVal) + offsetData;
	}
	else
	{
		return ((yVal * width + xVal) * totalNumDispVals + currentDisparity);
	}
}

template<typename T>
ARCHITECTURE_ADDITION inline T getZeroVal()
{
	return (T)0.0;
}

//avx512 requires data to be aligned on 64 bytes (16 float values)
#if CPU_OPTIMIZATION_SETTING == USE_AVX_512
#define DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT 16
#else
#define DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT 8
#endif

//inline function to check if data is aligned at xValDataStart for SIMD loads/stores that require alignment
inline bool MemoryAlignedAtDataStart(int xValDataStart, int numDataInSIMDVector)
{
	//assuming that the padded checkerboard width divides evenly by NUM_DATA_ALIGN_WIDTH_FROM_PYTHON (if that's not the case it's a bug)
	return (((xValDataStart % numDataInSIMDVector) == 0) && ((NUM_DATA_ALIGN_WIDTH_FROM_PYTHON % DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT) == 0));
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T>
ARCHITECTURE_ADDITION inline void dtStereo(T f[bp_params::NUM_POSSIBLE_DISPARITY_VALUES])
{
	T prev;
	for (int currentDisparity = 1; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = f[currentDisparity-1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = bp_params::NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity+1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}


template<typename T, typename U>
ARCHITECTURE_ADDITION inline void msgStereo(int xVal, int yVal, const levelProperties& currentLevelProperties, U messageValsNeighbor1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], U messageValsNeighbor2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
	U messageValsNeighbor3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], U dataCosts[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
	T* dstMessageArray, U disc_k_bp, bool dataAligned)
{
	// aggregate and find min
	U minimum = bp_consts::INF_BP;

	U dst[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereo<U>(dst);

	// truncate
	minimum += disc_k_bp;

	// normalize
	U valToNormalize = (U)0.0;

	for (int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}

	valToNormalize /= ((U)bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

	for (int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] -= valToNormalize;
		dstMessageArray[destMessageArrayIndex] = convertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
#ifdef _WIN32
	//assuming that width includes padding
	if constexpr (OPTIMIZED_INDEXING_SETTING)
#else
	if (OPTIMIZED_INDEXING_SETTING)
#endif
		{
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel;
		}
		else
		{
			destMessageArrayIndex++;
		}
	}
}


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T, typename U>
ARCHITECTURE_ADDITION inline void initializeBottomLevelDataStereoPixel(int xVal, int yVal,
		const levelProperties& currentLevelProperties, float* image1PixelsDevice,
		float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard0,
		T* dataCostDeviceStereoCheckerboard1, float lambda_bp,
		float data_k_bp)
{
	int indexVal;
	int xInCheckerboard = xVal / 2;

	if (withinImageBounds(xInCheckerboard, yVal,
			currentLevelProperties.widthCheckerboardLevel,
			currentLevelProperties.heightLevel)) {
		//make sure that it is possible to check every disparity value
		//need to cast bp_params::NUM_POSSIBLE_DISPARITY_VALUES from unsigned int to int
		//for conditional to work as expected
		if ((xVal - ((int)bp_params::NUM_POSSIBLE_DISPARITY_VALUES - 1)) >= 0) {
			for (int currentDisparity = 0;
					currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				float currentPixelImage1 = 0.0f;
				float currentPixelImage2 = 0.0f;

				if (withinImageBounds(xVal, yVal,
						currentLevelProperties.widthLevel,
						currentLevelProperties.heightLevel)) {
					currentPixelImage1 = image1PixelsDevice[yVal
							* currentLevelProperties.widthLevel + xVal];
					currentPixelImage2 = image2PixelsDevice[yVal
							* currentLevelProperties.widthLevel
							+ (xVal - currentDisparity)];
				}

				indexVal = retrieveIndexInDataAndMessage(xInCheckerboard,
						yVal,
						currentLevelProperties.paddedWidthCheckerboardLevel,
						currentLevelProperties.heightLevel, currentDisparity,
						bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0) {
					dataCostDeviceStereoCheckerboard0[indexVal] = convertValToDifferentDataTypeIfNeeded<float, T>((float) (lambda_bp
							* getMin<float>(
									((float) fabs(
											currentPixelImage1
													- currentPixelImage2)),
									data_k_bp)));
				} else {
					dataCostDeviceStereoCheckerboard1[indexVal] = convertValToDifferentDataTypeIfNeeded<float, T>((float) (lambda_bp
							* getMin<float>(
									((float) abs(
											currentPixelImage1
													- currentPixelImage2)),
									data_k_bp)));
				}
			}
		} else {
			for (int currentDisparity = 0;
					currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				indexVal = retrieveIndexInDataAndMessage(xInCheckerboard,
						yVal,
						currentLevelProperties.paddedWidthCheckerboardLevel,
						currentLevelProperties.heightLevel, currentDisparity,
						bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0) {
					dataCostDeviceStereoCheckerboard0[indexVal] = getZeroVal<
							T>();
				} else {
					dataCostDeviceStereoCheckerboard1[indexVal] = getZeroVal<
							T>();
				}
			}
		}
	}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T, typename U>
ARCHITECTURE_ADDITION inline void initializeCurrentLevelDataStereoPixel(
		int xVal, int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties, T* dataCostStereoCheckerboard0,
		T* dataCostStereoCheckerboard1, T* dataCostDeviceToWriteTo,
		int offsetNum) {
	//add 1 or 0 to the x-value depending on checkerboard part and row adding to; CHECKERBOARD_PART_0 with slot at (0, 0) has adjustment of 0 in row 0,
	//while CHECKERBOARD_PART_1 with slot at (0, 1) has adjustment of 1 in row 0
	int checkerboardPartAdjustment = 0;

	if (checkerboardPart == CHECKERBOARD_PART_0) {
		checkerboardPartAdjustment = (yVal % 2);
	} else /*(checkerboardPart == CHECKERBOARD_PART_1)*/ {
		checkerboardPartAdjustment = ((yVal + 1) % 2);
	}

	//the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
	int xValPrev = xVal * 2 + checkerboardPartAdjustment;

	if (withinImageBounds(xValPrev, (yVal * 2 + 1),
			prevLevelProperties.widthCheckerboardLevel,
			prevLevelProperties.heightLevel)) {
		for (int currentDisparity = 0;
				currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			U dataCostVal = convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
					xValPrev, (yVal * 2),
					prevLevelProperties.paddedWidthCheckerboardLevel,
					prevLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])
					+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xValPrev, (yVal * 2),
							prevLevelProperties.paddedWidthCheckerboardLevel,
							prevLevelProperties.heightLevel,
							currentDisparity,
							bp_params::NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])
					+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xValPrev, (yVal * 2 + 1),
							prevLevelProperties.paddedWidthCheckerboardLevel,
							prevLevelProperties.heightLevel,
							currentDisparity,
							bp_params::NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])
					+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xValPrev, (yVal * 2 + 1),
							prevLevelProperties.paddedWidthCheckerboardLevel,
							prevLevelProperties.heightLevel,
							currentDisparity,
							bp_params::NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)]);

			dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] =
							convertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
		}
	}
}

//initialize the message values at each pixel of the current level to the default value
template<typename T>
ARCHITECTURE_ADDITION inline void initializeMessageValsToDefaultKernelPixel(
		int xValInCheckerboard, int yVal,
		const levelProperties& currentLevelProperties,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1)
{
	//initialize message values in both checkerboards

	//set the message value at each pixel for each disparity to 0
	for (int currentDisparity = 0;
			currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
	}

	//retrieve the previous message value at each movement at each pixel
	for (int currentDisparity = 0;
			currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
	}
}



//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<typename T, typename U>
ARCHITECTURE_ADDITION inline void runBPIterationInOutDataInLocalMem(int xVal, int yVal, const levelProperties& currentLevelProperties, U prevUMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], U prevDMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], U prevLMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], U prevRMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], U dataMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
								T* currentUMessageArray, T* currentDMessageArray, T* currentLMessageArray, T* currentRMessageArray, U disc_k_bp, bool dataAligned)
{
	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessageArray, disc_k_bp, dataAligned);
}


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<typename T, typename U>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel(
		int xVal, int yVal, const Checkerboard_Parts checkerboardToUpdate,
		const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, float disc_k_bp,
		int offsetData, bool dataAligned)
{
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_0)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_1
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (currentLevelProperties.heightLevel - 1)))
	{
		U prevUMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];
		U prevDMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];
		U prevLMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];
		U prevRMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

		U dataMessage[bp_params::NUM_POSSIBLE_DISPARITY_VALUES];

		for (int currentDisparity = 0; currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_0)
			{
				dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_1
			{
				dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}

		//uses the previous message values and data cost to calculate the current message values and store the results
		if (checkerboardToUpdate == CHECKERBOARD_PART_0)
		{
			runBPIterationInOutDataInLocalMem<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage,
					prevLMessage, prevRMessage, dataMessage,
					messageUDeviceCurrentCheckerboard0,
					messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0,
					messageRDeviceCurrentCheckerboard0, (U) disc_k_bp,
					dataAligned);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_1
		{
			runBPIterationInOutDataInLocalMem<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage,
					prevLMessage, prevRMessage, dataMessage,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1, (U) disc_k_bp,
					dataAligned);
		}
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T>
ARCHITECTURE_ADDITION inline void copyPrevLevelToNextLevelBPCheckerboardStereoPixel(int xVal, int yVal,
		const Checkerboard_Parts checkerboardPart, const levelProperties& currentLevelProperties,
		const levelProperties& nextLevelProperties,
		T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
		T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1) {
	int indexCopyTo;
	int indexCopyFrom;

	int checkerboardPartAdjustment;

	T prevValU;
	T prevValD;
	T prevValL;
	T prevValR;

	if (checkerboardPart == CHECKERBOARD_PART_0) {
		checkerboardPartAdjustment = (yVal % 2);
	} else /*(checkerboardPart == CHECKERBOARD_PART_1)*/ {
		checkerboardPartAdjustment = ((yVal + 1) % 2);
	}

	for (int currentDisparity = 0;
			currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

		if (checkerboardPart == CHECKERBOARD_PART_0) {
			prevValU = messageUPrevStereoCheckerboard0[indexCopyFrom];
			prevValD = messageDPrevStereoCheckerboard0[indexCopyFrom];
			prevValL = messageLPrevStereoCheckerboard0[indexCopyFrom];
			prevValR = messageRPrevStereoCheckerboard0[indexCopyFrom];
		} else /*(checkerboardPart == CHECKERBOARD_PART_1)*/ {
			prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
			prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
			prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
			prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
		}

		if (withinImageBounds(xVal * 2 + checkerboardPartAdjustment,
				yVal * 2, nextLevelProperties.widthCheckerboardLevel,
				nextLevelProperties.heightLevel)) {
			indexCopyTo = retrieveIndexInDataAndMessage(
					(xVal * 2 + checkerboardPartAdjustment), (yVal * 2),
					nextLevelProperties.paddedWidthCheckerboardLevel,
					nextLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

			messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

			messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
		}

		if (withinImageBounds(xVal * 2 + checkerboardPartAdjustment,
				yVal * 2 + 1, nextLevelProperties.widthCheckerboardLevel,
				nextLevelProperties.heightLevel)) {
			indexCopyTo = retrieveIndexInDataAndMessage(
					(xVal * 2 + checkerboardPartAdjustment), (yVal * 2 + 1),
					nextLevelProperties.paddedWidthCheckerboardLevel,
					nextLevelProperties.heightLevel, currentDisparity,
					bp_params::NUM_POSSIBLE_DISPARITY_VALUES);

			messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

			messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
		}
	}
}

template<typename T, typename U>
ARCHITECTURE_ADDITION inline void retrieveOutputDisparityCheckerboardStereoOptimizedPixel(int xVal, int yVal,
		const levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard0,
		T* dataCostStereoCheckerboard1, T* messageUPrevStereoCheckerboard0,
		T* messageDPrevStereoCheckerboard0, T* messageLPrevStereoCheckerboard0,
		T* messageRPrevStereoCheckerboard0, T* messageUPrevStereoCheckerboard1,
		T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1,
		T* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice) {

	int xValInCheckerboardPart = xVal;

	//first processing from first part of checkerboard
	{
		//adjustment based on checkerboard; need to add 1 to x for odd-numbered rows for final index mapping into disparity images for checkerboard 1
		int checkerboardPartAdjustment = (yVal % 2);

		if (withinImageBounds(
				xValInCheckerboardPart * 2 + checkerboardPartAdjustment, yVal,
				currentLevelProperties.widthLevel,
				currentLevelProperties.heightLevel)) {
			if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment))
					&& (xValInCheckerboardPart
							< (currentLevelProperties.widthCheckerboardLevel
									- checkerboardPartAdjustment)) && (yVal > 0)
					&& (yVal < (currentLevelProperties.heightLevel - 1))) {
				// keep track of "best" disparity for current pixel
				int bestDisparity = 0;
				U best_val = bp_consts::INF_BP;
				for (int currentDisparity = 0;
						currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					U val =
							convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal + 1),
									currentLevelProperties.paddedWidthCheckerboardLevel,
									currentLevelProperties.heightLevel,
									currentDisparity,
									bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, (yVal - 1),
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart - 1
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);

					if (val < (best_val)) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}

				disparityBetweenImagesDevice[yVal
						* currentLevelProperties.widthLevel
						+ (xValInCheckerboardPart * 2
								+ checkerboardPartAdjustment)] = bestDisparity;
			} else {
				disparityBetweenImagesDevice[yVal
						* currentLevelProperties.widthLevel
						+ (xValInCheckerboardPart * 2
								+ checkerboardPartAdjustment)] = 0;
			}
		}
	}
	//process from part 2 of checkerboard
	{
		//adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
		int checkerboardPartAdjustment = ((yVal + 1) % 2);

		if (withinImageBounds(
				xValInCheckerboardPart * 2 + checkerboardPartAdjustment, yVal,
				currentLevelProperties.widthLevel,
				currentLevelProperties.heightLevel)) {
			if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment))
					&& (xValInCheckerboardPart
							< (currentLevelProperties.widthCheckerboardLevel
									- checkerboardPartAdjustment)) && (yVal > 0)
					&& (yVal < (currentLevelProperties.heightLevel - 1))) {
				// keep track of "best" disparity for current pixel
				int bestDisparity = 0;
				U best_val = bp_consts::INF_BP;
				for (int currentDisparity = 0;
						currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					U val =
							convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal + 1),
									currentLevelProperties.paddedWidthCheckerboardLevel,
									currentLevelProperties.heightLevel,
									currentDisparity,
									bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, (yVal - 1),
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart - 1
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);

					if (val < (best_val)) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
				disparityBetweenImagesDevice[yVal
						* currentLevelProperties.widthLevel
						+ (xValInCheckerboardPart * 2
								+ checkerboardPartAdjustment)] = bestDisparity;
			} else {
				disparityBetweenImagesDevice[yVal
						* currentLevelProperties.widthLevel
						+ (xValInCheckerboardPart * 2
								+ checkerboardPartAdjustment)] = 0;
			}
		}
	}
}


template<typename T>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsAtPointKernel(int xVal, int yVal, const levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}

template<typename T>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsToPointKernel(int xVal, int yVal, const levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0,
		T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0,
		T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1)
{
	int checkerboardAdjustment;
	if (((xVal + yVal) % 2) == 0)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_1
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}
	if (((xVal + yVal) % 2) == 0) {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		} else {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < bp_params::NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, bp_params::NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}
}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
