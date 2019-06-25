/*
 * SharedBPProcessingFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDBPPROCESSINGFUNCTS_H_
#define SHAREDBPPROCESSINGFUNCTS_H_

//indexing is performed in such a way so that the memory accesses as coalesced as much as possible
#if OPTIMIZED_INDEXING_SETTING == 1
#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU (yVal*width*totalNumDispVals + width*currentDisparity + xVal)
#else
#define RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU ((yVal*width + xVal)*totalNumDispVals + currentDisparity)
#endif

template<typename T, typename U>
inline U convertValToDifferentDataTypeIfNeeded(T data)
{
	//by default assume same data type and just return data
	return data;
}

//checks if the current point is within the image bounds
inline bool withinImageBounds(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
inline int retrieveIndexInDataAndMessage(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData = 0)
{
	//assuming that width includes padding
	return RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU + offsetData;
}

template<typename T>
inline T getZeroVal()
{
	return (T)0.0;
}



//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T>
inline void dtStereo(T f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	T prev;
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = f[currentDisparity-1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity+1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}


template<typename T>
inline void msgStereo(T messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], T messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
	T messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], T dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
	T dst[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp)
{
	// aggregate and find min
	T minimum = INF_BP;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereo<T>(dst);

	// truncate
	minimum += disc_k_bp;

	// normalize
	T valToNormalize = 0;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}

	valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		dst[currentDisparity] -= valToNormalize;
}


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T, typename U>
inline void initializeBottomLevelDataStereoPixel(int xVal, int yVal,
		levelProperties& currentLevelProperties, float* image1PixelsDevice,
		float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard1,
		T* dataCostDeviceStereoCheckerboard2, float lambda_bp,
		float data_k_bp)
{
	int indexVal;
	int xInCheckerboard = xVal / 2;

	if (withinImageBounds(xInCheckerboard, yVal,
			currentLevelProperties.widthCheckerboardLevel,
			currentLevelProperties.heightLevel)) {
		//make sure that it is possible to check every disparity value
		if ((xVal - (NUM_POSSIBLE_DISPARITY_VALUES - 1)) >= 0) {
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
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
						NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0) {
					dataCostDeviceStereoCheckerboard1[indexVal] = convertValToDifferentDataTypeIfNeeded<U, T>((U) (lambda_bp
							* std::min(
									((U) abs(
											currentPixelImage1
													- currentPixelImage2)),
									(U) data_k_bp)));
				} else {
					dataCostDeviceStereoCheckerboard2[indexVal] = convertValToDifferentDataTypeIfNeeded<U, T>((U) (lambda_bp
							* std::min(
									((U) abs(
											currentPixelImage1
													- currentPixelImage2)),
									(U) data_k_bp)));
				}
			}
		} else {
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				indexVal = retrieveIndexInDataAndMessage(xInCheckerboard,
						yVal,
						currentLevelProperties.paddedWidthCheckerboardLevel,
						currentLevelProperties.heightLevel, currentDisparity,
						NUM_POSSIBLE_DISPARITY_VALUES);

				//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
				if (((xVal + yVal) % 2) == 0) {
					dataCostDeviceStereoCheckerboard1[indexVal] = getZeroVal<
							T>();
				} else {
					dataCostDeviceStereoCheckerboard2[indexVal] = getZeroVal<
							T>();
				}
			}
		}
	}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T, typename U>
inline void initializeCurrentLevelDataStereoNoTexturesPixel(
		int xVal, int yVal, int checkerboardPart,
		levelProperties& currentLevelProperties,
		levelProperties& prevLevelProperties, T* dataCostStereoCheckerboard1,
		T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteTo,
		int offsetNum) {
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

	if (withinImageBounds(xValPrev, (yVal * 2 + 1),
			prevLevelProperties.widthCheckerboardLevel,
			prevLevelProperties.heightLevel)) {
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			convertValToDifferentDataTypeIfNeeded<U, T>(dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel,
					currentLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES)] =
							convertValToDifferentDataTypeIfNeeded<T, U>((dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xValPrev, (yVal * 2),
							prevLevelProperties.paddedWidthCheckerboardLevel,
							prevLevelProperties.heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])
							+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
									xValPrev, (yVal * 2),
									prevLevelProperties.paddedWidthCheckerboardLevel,
									prevLevelProperties.heightLevel,
									currentDisparity,
									NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])
							+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
									xValPrev, (yVal * 2 + 1),
									prevLevelProperties.paddedWidthCheckerboardLevel,
									prevLevelProperties.heightLevel,
									currentDisparity,
									NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])
							+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
									xValPrev, (yVal * 2 + 1),
									prevLevelProperties.paddedWidthCheckerboardLevel,
									prevLevelProperties.heightLevel,
									currentDisparity,
									NUM_POSSIBLE_DISPARITY_VALUES, offsetNum)])));
		}
	}
}

//initialize the message values at each pixel of the current level to the default value
template<typename T>
inline void initializeMessageValsToDefaultKernelPixel(int xValInCheckerboard,  int yVal, levelProperties& currentLevelProperties, T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2)
{
	//initialize message values in both checkerboards

	//set the message value at each pixel for each disparity to 0
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
	}

	//retrieve the previous message value at each movement at each pixel
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
		messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
				xValInCheckerboard, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroVal<T>();
	}
}



//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<typename T>
inline void runBPIterationInOutDataInLocalMem(T prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T dataMessage[NUM_POSSIBLE_DISPARITY_VALUES],
								T currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp)
{
	msgStereo<T>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessage, disc_k_bp);

	msgStereo<T>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessage, disc_k_bp);

	msgStereo<T>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessage, disc_k_bp);

	msgStereo<T>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessage, disc_k_bp);
}


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<typename T, typename U>
inline void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel(
		int xVal, int yVal, int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
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
		U prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		U prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		U prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		U prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		U dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}

		U currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		U currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		U currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		U currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		//uses the previous message values and data cost to calculate the current message values and store the results
		runBPIterationInOutDataInLocalMem<U>(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							currentUMessage, currentDMessage, currentLMessage, currentRMessage, (T)disc_k_bp);

		//write the calculated message values to global memory
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				messageUDeviceCurrentCheckerboard1[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentUMessage[currentDisparity]);
				messageDDeviceCurrentCheckerboard1[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentDMessage[currentDisparity]);
				messageLDeviceCurrentCheckerboard1[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentLMessage[currentDisparity]);
				messageRDeviceCurrentCheckerboard1[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentRMessage[currentDisparity]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				messageUDeviceCurrentCheckerboard2[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentUMessage[currentDisparity]);
				messageDDeviceCurrentCheckerboard2[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentDMessage[currentDisparity]);
				messageLDeviceCurrentCheckerboard2[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentLMessage[currentDisparity]);
				messageRDeviceCurrentCheckerboard2[indexWriteTo] = convertValToDifferentDataTypeIfNeeded<U, T>(currentRMessage[currentDisparity]);
			}
		}
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T>
inline void copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesPixel(int xVal, int yVal,
		int checkerboardPart, levelProperties& currentLevelProperties,
		levelProperties& nextLevelProperties,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2,
		T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2) {
	int indexCopyTo;
	int indexCopyFrom;

	int checkerboardPartAdjustment;

	T prevValU;
	T prevValD;
	T prevValL;
	T prevValR;

	if (checkerboardPart == CHECKERBOARD_PART_1) {
		checkerboardPartAdjustment = (yVal % 2);
	} else if (checkerboardPart == CHECKERBOARD_PART_2) {
		checkerboardPartAdjustment = ((yVal + 1) % 2);
	}

	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, currentDisparity,
				NUM_POSSIBLE_DISPARITY_VALUES);

		if (checkerboardPart == CHECKERBOARD_PART_1) {
			prevValU = messageUPrevStereoCheckerboard1[indexCopyFrom];
			prevValD = messageDPrevStereoCheckerboard1[indexCopyFrom];
			prevValL = messageLPrevStereoCheckerboard1[indexCopyFrom];
			prevValR = messageRPrevStereoCheckerboard1[indexCopyFrom];
		} else if (checkerboardPart == CHECKERBOARD_PART_2) {
			prevValU = messageUPrevStereoCheckerboard2[indexCopyFrom];
			prevValD = messageDPrevStereoCheckerboard2[indexCopyFrom];
			prevValL = messageLPrevStereoCheckerboard2[indexCopyFrom];
			prevValR = messageRPrevStereoCheckerboard2[indexCopyFrom];
		}

		if (withinImageBounds(xVal * 2 + checkerboardPartAdjustment,
				yVal * 2, nextLevelProperties.widthCheckerboardLevel,
				nextLevelProperties.heightLevel)) {
			indexCopyTo = retrieveIndexInDataAndMessage(
					(xVal * 2 + checkerboardPartAdjustment), (yVal * 2),
					nextLevelProperties.paddedWidthCheckerboardLevel,
					nextLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES);

			messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;

			messageUDeviceCurrentCheckerboard2[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard2[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard2[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard2[indexCopyTo] = prevValR;
		}

		if (withinImageBounds(xVal * 2 + checkerboardPartAdjustment,
				yVal * 2 + 1, nextLevelProperties.widthCheckerboardLevel,
				nextLevelProperties.heightLevel)) {
			indexCopyTo = retrieveIndexInDataAndMessage(
					(xVal * 2 + checkerboardPartAdjustment), (yVal * 2 + 1),
					nextLevelProperties.paddedWidthCheckerboardLevel,
					nextLevelProperties.heightLevel, currentDisparity,
					NUM_POSSIBLE_DISPARITY_VALUES);

			messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;

			messageUDeviceCurrentCheckerboard2[indexCopyTo] = prevValU;
			messageDDeviceCurrentCheckerboard2[indexCopyTo] = prevValD;
			messageLDeviceCurrentCheckerboard2[indexCopyTo] = prevValL;
			messageRDeviceCurrentCheckerboard2[indexCopyTo] = prevValR;
		}
	}
}

template<typename T, typename U>
inline void retrieveOutputDisparityCheckerboardStereoOptimizedPixel(int xVal, int yVal,
		levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1,
		T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1,
		T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1,
		T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2,
		T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2,
		T* messageRPrevStereoCheckerboard2,
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
				U best_val = INF_BP;
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					U val =
							convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal + 1),
									currentLevelProperties.paddedWidthCheckerboardLevel,
									currentLevelProperties.heightLevel,
									currentDisparity,
									NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, (yVal - 1),
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard2[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart - 1
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);

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
				U best_val = INF_BP;
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					U val =
							convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal + 1),
									currentLevelProperties.paddedWidthCheckerboardLevel,
									currentLevelProperties.heightLevel,
									currentDisparity,
									NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, (yVal - 1),
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
											(xValInCheckerboardPart - 1
													+ checkerboardPartAdjustment),
											yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)])
									+ convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
											xValInCheckerboardPart, yVal,
											currentLevelProperties.paddedWidthCheckerboardLevel,
											currentLevelProperties.heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]);

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
inline void printDataAndMessageValsAtPointKernel(int xVal, int yVal, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}

template<typename T>
inline void printDataAndMessageValsToPointKernel(int xVal, int yVal, levelProperties& currentLevelProperties, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2)
{
	int checkerboardAdjustment;
	if (((xVal + yVal) % 2) == 0)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}
	if (((xVal + yVal) % 2) == 0) {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		} else {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(
								xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}
}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
