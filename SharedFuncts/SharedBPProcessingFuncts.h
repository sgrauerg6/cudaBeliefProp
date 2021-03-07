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
ARCHITECTURE_ADDITION inline U convertValToDifferentDataTypeIfNeeded(const T data) {
	return data; //by default assume same data type and just return data
}

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
ARCHITECTURE_ADDITION inline unsigned int retrieveIndexInDataAndMessage(const unsigned int xVal, const unsigned int yVal,
		const unsigned int width, const unsigned int height, const unsigned int currentDisparity, const unsigned int totalNumDispVals,
		const unsigned int offsetData = 0u)
{
#ifdef _WIN32
	//assuming that width includes padding
	if /*constexpr*/ (OPTIMIZED_INDEXING_SETTING)
#else
	if (OPTIMIZED_INDEXING_SETTING)
#endif
	{
		//indexing is performed in such a way so that the memory accesses as coalesced as much as possible
		return (yVal * width * totalNumDispVals + width * currentDisparity + xVal) + offsetData;
	}
	else {
		return ((yVal * width + xVal) * totalNumDispVals + currentDisparity);
	}
}

template<typename T>
ARCHITECTURE_ADDITION inline T getZeroVal() {
	return (T)0.0;
}

//avx512 requires data to be aligned on 64 bytes (16 float values)
#if CPU_OPTIMIZATION_SETTING == USE_AVX_512
#define DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT 16
#else
#define DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT 8
#endif

//inline function to check if data is aligned at xValDataStart for SIMD loads/stores that require alignment
inline bool MemoryAlignedAtDataStart(const unsigned int xValDataStart, const unsigned int numDataInSIMDVector)
{
	//assuming that the padded checkerboard width divides evenly by NUM_DATA_ALIGN_WIDTH_FROM_PYTHON (if that's not the case it's a bug)
	return (((xValDataStart % numDataInSIMDVector) == 0) && ((NUM_DATA_ALIGN_WIDTH_FROM_PYTHON % DIVISOR_FOR_PADDED_CHECKERBOARD_WIDTH_FOR_ALIGNMENT) == 0));
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void dtStereo(T f[DISP_VALS])
{
	T prev;
	for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
	{
		prev = f[currentDisparity-1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = (int)DISP_VALS - 2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity + 1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}

template<typename T>
ARCHITECTURE_ADDITION inline void dtStereo(T*& f, const unsigned int bpSettingsDispVals)
{
	T prev;
	for (unsigned int currentDisparity = 1; currentDisparity < bpSettingsDispVals; currentDisparity++)
	{
		prev = f[currentDisparity-1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = (int)bpSettingsDispVals - 2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity + 1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}


template<typename T, typename U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void msgStereo(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		U messageValsNeighbor1[DISP_VALS], U messageValsNeighbor2[DISP_VALS],
		U messageValsNeighbor3[DISP_VALS], U dataCosts[DISP_VALS],
		T* dstMessageArray, U disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	U minimum{(U)bp_consts::INF_BP};
	U dst[DISP_VALS];

	for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereo<U, DISP_VALS>(dst);

	// truncate
	minimum += disc_k_bp;

	// normalize
	U valToNormalize{(U)0.0};

	for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
		if (minimum < dst[currentDisparity]) {
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}

	valToNormalize /= ((U)DISP_VALS);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, 0,
			DISP_VALS);

	for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
		dst[currentDisparity] -= valToNormalize;
		dstMessageArray[destMessageArrayIndex] = convertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
#ifdef _WIN32
	//assuming that width includes padding
	if /*constexpr*/ (OPTIMIZED_INDEXING_SETTING)
#else
	if (OPTIMIZED_INDEXING_SETTING)
#endif
		{
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
		}
		else {
			destMessageArrayIndex++;
		}
	}
}


template<typename T, typename U>
ARCHITECTURE_ADDITION inline void msgStereo(const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		U*& messageValsNeighbor1, U*& messageValsNeighbor2,
		U*& messageValsNeighbor3, U*& dataCosts,
		T* dstMessageArray, U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	// aggregate and find min
	U minimum{(U)bp_consts::INF_BP};
	U* dst = new U[bpSettingsDispVals];

	for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereo<U>(dst, bpSettingsDispVals);

	// truncate
	minimum += disc_k_bp;

	// normalize
	U valToNormalize{(U)0.0};

	for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
		if (minimum < dst[currentDisparity]) {
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}

	valToNormalize /= ((U)bpSettingsDispVals);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, 0,
			bpSettingsDispVals);

	for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
		dst[currentDisparity] -= valToNormalize;
		dstMessageArray[destMessageArrayIndex] = convertValToDifferentDataTypeIfNeeded<U, T>(dst[currentDisparity]);
#ifdef _WIN32
	//assuming that width includes padding
	if /*constexpr*/ (OPTIMIZED_INDEXING_SETTING)
#else
	if (OPTIMIZED_INDEXING_SETTING)
#endif
		{
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
		}
		else {
			destMessageArrayIndex++;
		}
	}

	delete [] dst;
}


//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeBottomLevelDataStereoPixel(const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties, float* image1PixelsDevice,
		float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard0,
		T* dataCostDeviceStereoCheckerboard1, const float lambda_bp,
		const float data_k_bp, const unsigned int bpSettingsDispVals)
{
	if constexpr (DISP_VALS > 0) {
		unsigned int indexVal;
		const unsigned int xInCheckerboard = xVal / 2;

		if (withinImageBounds(xInCheckerboard, yVal, currentLevelProperties.widthCheckerboardLevel_, currentLevelProperties.heightLevel_)) {
			//make sure that it is possible to check every disparity value
			//need to cast DISP_VALS from unsigned int to int
			//for conditional to work as expected
			if (((int)xVal - ((int)DISP_VALS - 1)) >= 0) {
				for (unsigned int currentDisparity = 0u; currentDisparity < DISP_VALS; currentDisparity++) {
					float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

					if (withinImageBounds(xVal, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
						currentPixelImage1 = image1PixelsDevice[yVal * currentLevelProperties.widthLevel_ + xVal];
						currentPixelImage2 = image2PixelsDevice[yVal * currentLevelProperties.widthLevel_ + (xVal - currentDisparity)];
					}

					indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel_,
							currentLevelProperties.heightLevel_, currentDisparity,
							DISP_VALS);

					//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
					if (((xVal + yVal) % 2) == 0) {
						dataCostDeviceStereoCheckerboard0[indexVal] = convertValToDifferentDataTypeIfNeeded<float, T>(
								(float)(lambda_bp * getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
					}
					else {
						dataCostDeviceStereoCheckerboard1[indexVal] = convertValToDifferentDataTypeIfNeeded<float, T>(
								(float)(lambda_bp * getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
					}
				}
			} else {
				for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
					indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS);

					//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
					if (((xVal + yVal) % 2) == 0) {
						dataCostDeviceStereoCheckerboard0[indexVal] = getZeroVal<T>();
					}
					else {
						dataCostDeviceStereoCheckerboard1[indexVal] = getZeroVal<T>();
					}
				}
			}
		}
	}
	else {
		unsigned int indexVal;
		const unsigned int xInCheckerboard = xVal / 2;

		if (withinImageBounds(xInCheckerboard, yVal, currentLevelProperties.widthCheckerboardLevel_, currentLevelProperties.heightLevel_)) {
			//make sure that it is possible to check every disparity value
			//need to cast bpSettingsDispVals from unsigned int to int
			//for conditional to work as expected
			if (((int)xVal - ((int)bpSettingsDispVals - 1)) >= 0) {
				for (unsigned int currentDisparity = 0u; currentDisparity < bpSettingsDispVals; currentDisparity++) {
					float currentPixelImage1 = 0.0f, currentPixelImage2 = 0.0f;

					if (withinImageBounds(xVal, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
						currentPixelImage1 = image1PixelsDevice[yVal * currentLevelProperties.widthLevel_ + xVal];
						currentPixelImage2 = image2PixelsDevice[yVal * currentLevelProperties.widthLevel_ + (xVal - currentDisparity)];
					}

					indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel_,
							currentLevelProperties.heightLevel_, currentDisparity,
							bpSettingsDispVals);

					//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
					if (((xVal + yVal) % 2) == 0) {
						dataCostDeviceStereoCheckerboard0[indexVal] = convertValToDifferentDataTypeIfNeeded<float, T>(
								(float)(lambda_bp * getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
					}
					else {
						dataCostDeviceStereoCheckerboard1[indexVal] = convertValToDifferentDataTypeIfNeeded<float, T>(
								(float)(lambda_bp * getMin<float>((fabs(currentPixelImage1 - currentPixelImage2)), data_k_bp)));
					}
				}
			} else {
				for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
					indexVal = retrieveIndexInDataAndMessage(xInCheckerboard, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, bpSettingsDispVals);

					//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
					if (((xVal + yVal) % 2) == 0) {
						dataCostDeviceStereoCheckerboard0[indexVal] = getZeroVal<T>();
					}
					else {
						dataCostDeviceStereoCheckerboard1[indexVal] = getZeroVal<T>();
					}
				}
			}
		}
	}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T, typename U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeCurrentLevelDataStereoPixel(
		const unsigned int xVal, const unsigned int yVal, const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& prevLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* dataCostDeviceToWriteTo, const unsigned int offsetNum,
		const unsigned int bpSettingsDispVals)
{
	//add 1 or 0 to the x-value depending on checkerboard part and row adding to; CHECKERBOARD_PART_0 with slot at (0, 0) has adjustment of 0 in row 0,
	//while CHECKERBOARD_PART_1 with slot at (0, 1) has adjustment of 1 in row 0
	const unsigned int checkerboardPartAdjustment = (checkerboardPart == CHECKERBOARD_PART_0) ? (yVal % 2) : ((yVal + 1) % 2);

	//the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
	const unsigned int xValPrev = xVal*2 + checkerboardPartAdjustment;

	if (withinImageBounds(xValPrev, (yVal * 2 + 1), prevLevelProperties.widthCheckerboardLevel_, prevLevelProperties.heightLevel_)) {
		if constexpr (DISP_VALS > 0) {
			for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
				const U dataCostVal =
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, DISP_VALS, offsetNum)]) +
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, DISP_VALS, offsetNum)]) +
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, DISP_VALS, offsetNum)]) +
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, DISP_VALS, offsetNum)]);

				dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
						currentLevelProperties.paddedWidthCheckerboardLevel_,
						currentLevelProperties.heightLevel_, currentDisparity,
						DISP_VALS)] =
								convertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
			}
		}
		else {
			for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
				const U dataCostVal =
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, bpSettingsDispVals, offsetNum)]) +
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, bpSettingsDispVals, offsetNum)]) +
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, bpSettingsDispVals, offsetNum)]) +
						convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xValPrev, yVal*2 + 1, prevLevelProperties.paddedWidthCheckerboardLevel_, prevLevelProperties.heightLevel_,
								currentDisparity, bpSettingsDispVals, offsetNum)]);

				dataCostDeviceToWriteTo[retrieveIndexInDataAndMessage(xVal, yVal,
						currentLevelProperties.paddedWidthCheckerboardLevel_,
						currentLevelProperties.heightLevel_, currentDisparity,
						bpSettingsDispVals)] =
								convertValToDifferentDataTypeIfNeeded<U, T>(dataCostVal);
			}
		}
	}
}

//initialize the message values at each pixel of the current level to the default value
template<typename T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void initializeMessageValsToDefaultKernelPixel(
		const unsigned int xValInCheckerboard, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const unsigned int bpSettingsDispVals)
{
	//initialize message values in both checkerboards

	if constexpr (DISP_VALS > 0) {
		//set the message value at each pixel for each disparity to 0
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
			messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
			messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
			messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
		}

		//retrieve the previous message value at each movement at each pixel
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
			messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
			messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
			messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS)] =
							getZeroVal<T>();
		}
	}
	else {
		//set the message value at each pixel for each disparity to 0
		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
			messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
			messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
			messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
			messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
		}

		//retrieve the previous message value at each movement at each pixel
		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
			messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
			messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
			messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
			messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xValInCheckerboard, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals)] =
							getZeroVal<T>();
		}
	}
}


//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<typename T, typename U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationInOutDataInLocalMem(
		const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
		U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS],
		U dataMessage[DISP_VALS],
		T* currentUMessageArray, T* currentDMessageArray,
		T* currentLMessageArray, T* currentRMessageArray,
		const U disc_k_bp, const bool dataAligned)
{
	msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessageArray, disc_k_bp, dataAligned);
}


template<typename T, typename U>
ARCHITECTURE_ADDITION inline void runBPIterationInOutDataInLocalMem(
		const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		U*& prevUMessage, U*& prevDMessage,
		U*& prevLMessage, U*& prevRMessage,
		U*& dataMessage,
		T* currentUMessageArray, T* currentDMessageArray,
		T* currentLMessageArray, T* currentRMessageArray,
		const U disc_k_bp, const bool dataAligned, const unsigned int bpSettingsDispVals)
{
	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessageArray, disc_k_bp, dataAligned, bpSettingsDispVals);
}


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<typename T, typename U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel(
		const unsigned int xVal, const unsigned int yVal,
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int offsetData, const bool dataAligned,
		const unsigned int bpSettingsDispVals)
{
	//checkerboardAdjustment used for indexing into current checkerboard to update
	const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal)%2) : ((yVal+1)%2);

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	if ((xVal >= (1u - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment)) &&
		(yVal > 0) && (yVal < (currentLevelProperties.heightLevel_ - 1u)))
	{
		if constexpr (DISP_VALS > 0) {
			U dataMessage[DISP_VALS], prevUMessage[DISP_VALS], prevDMessage[DISP_VALS], prevLMessage[DISP_VALS], prevRMessage[DISP_VALS];

			for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS, offsetData)]);
					prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
					prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
					prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
					prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
				}
				else { //checkerboardToUpdate == CHECKERBOARD_PART_1
					dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS, offsetData)]);
					prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
					prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
					prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
					prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, DISP_VALS)]);
				}
			}

			//uses the previous message values and data cost to calculate the current message values and store the results
			if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
				runBPIterationInOutDataInLocalMem<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties,
						prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
						messageUDeviceCurrentCheckerboard0,	messageDDeviceCurrentCheckerboard0,
						messageLDeviceCurrentCheckerboard0,	messageRDeviceCurrentCheckerboard0,
						(U)disc_k_bp, dataAligned);
			}
			else { //checkerboardToUpdate == CHECKERBOARD_PART_1
				runBPIterationInOutDataInLocalMem<T, U, DISP_VALS>(xVal, yVal, currentLevelProperties,
						prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
						messageUDeviceCurrentCheckerboard1,	messageDDeviceCurrentCheckerboard1,
						messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
						(U)disc_k_bp, dataAligned);
			}
		}
		else {
			U* dataMessage = (U*)malloc(sizeof(U) * bpSettingsDispVals);
			U* prevUMessage = (U*)malloc(sizeof(U) * bpSettingsDispVals);
			U* prevDMessage = (U*)malloc(sizeof(U) * bpSettingsDispVals);
			U* prevLMessage = (U*)malloc(sizeof(U) * bpSettingsDispVals);
			U* prevRMessage = (U*)malloc(sizeof(U) * bpSettingsDispVals);

			for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(xVal, yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals, offsetData)]);
					prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
					prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
					prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
					prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
				}
				else { //checkerboardToUpdate == CHECKERBOARD_PART_1
					dataMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals, offsetData)]);
					prevUMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal+1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
					prevDMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(xVal, (yVal-1),
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
					prevLMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
					prevRMessage[currentDisparity] = convertValToDifferentDataTypeIfNeeded<T, U>(
							messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(((xVal + checkerboardAdjustment) - 1), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
									currentDisparity, bpSettingsDispVals)]);
				}
			}

			//uses the previous message values and data cost to calculate the current message values and store the results
			if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
				runBPIterationInOutDataInLocalMem<T, U>(xVal, yVal, currentLevelProperties,
						prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
						messageUDeviceCurrentCheckerboard0,	messageDDeviceCurrentCheckerboard0,
						messageLDeviceCurrentCheckerboard0,	messageRDeviceCurrentCheckerboard0,
						(U)disc_k_bp, dataAligned, bpSettingsDispVals);
			}
			else { //checkerboardToUpdate == CHECKERBOARD_PART_1
				runBPIterationInOutDataInLocalMem<T, U>(xVal, yVal, currentLevelProperties,
						prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
						messageUDeviceCurrentCheckerboard1,	messageDDeviceCurrentCheckerboard1,
						messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
						(U)disc_k_bp, dataAligned, bpSettingsDispVals);
			}

			free(dataMessage);
			free(prevUMessage);
			free(prevDMessage);
			free(prevLMessage);
			free(prevRMessage);
		}
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void copyPrevLevelToNextLevelBPCheckerboardStereoPixel(
		const unsigned int xVal, const unsigned int yVal,
		const Checkerboard_Parts checkerboardPart, const levelProperties& currentLevelProperties,
		const levelProperties& nextLevelProperties,
		T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
		T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const unsigned int bpSettingsDispVals)
{
	unsigned int indexCopyTo, indexCopyFrom;
	T prevValU, prevValD, prevValL, prevValR;
	const unsigned int checkerboardPartAdjustment = (checkerboardPart == CHECKERBOARD_PART_0) ? (yVal % 2) : ((yVal + 1) % 2);

	if constexpr (DISP_VALS > 0) {
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, DISP_VALS);

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

			if (withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
				indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2,
						nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
						currentDisparity, DISP_VALS);

				messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
				messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
				messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
				messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

				messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
				messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
				messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
				messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
			}

			if (withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
				indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1,
						nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
						currentDisparity, DISP_VALS);

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
	else {
		for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
			indexCopyFrom = retrieveIndexInDataAndMessage(xVal, yVal,
					currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
					currentDisparity, bpSettingsDispVals);

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

			if (withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
				indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2,
						nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
						currentDisparity, bpSettingsDispVals);

				messageUDeviceCurrentCheckerboard0[indexCopyTo] = prevValU;
				messageDDeviceCurrentCheckerboard0[indexCopyTo] = prevValD;
				messageLDeviceCurrentCheckerboard0[indexCopyTo] = prevValL;
				messageRDeviceCurrentCheckerboard0[indexCopyTo] = prevValR;

				messageUDeviceCurrentCheckerboard1[indexCopyTo] = prevValU;
				messageDDeviceCurrentCheckerboard1[indexCopyTo] = prevValD;
				messageLDeviceCurrentCheckerboard1[indexCopyTo] = prevValL;
				messageRDeviceCurrentCheckerboard1[indexCopyTo] = prevValR;
			}

			if (withinImageBounds(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1, nextLevelProperties.widthCheckerboardLevel_, nextLevelProperties.heightLevel_)) {
				indexCopyTo = retrieveIndexInDataAndMessage(xVal*2 + checkerboardPartAdjustment, yVal*2 + 1,
						nextLevelProperties.paddedWidthCheckerboardLevel_, nextLevelProperties.heightLevel_,
						currentDisparity, bpSettingsDispVals);

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
}

template<typename T, typename U, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void retrieveOutputDisparityCheckerboardStereoOptimizedPixel(
		const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUPrevStereoCheckerboard0,	T* messageDPrevStereoCheckerboard0,
		T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals)
{
	const unsigned int xValInCheckerboardPart = xVal;

	//first processing from first part of checkerboard

	//adjustment based on checkerboard; need to add 1 to x for odd-numbered rows for final index mapping into disparity images for checkerboard 1
	unsigned int checkerboardPartAdjustment = (yVal % 2);

	if (withinImageBounds(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
		if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) &&
			(xValInCheckerboardPart < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardPartAdjustment)) &&
			(yVal > 0u) && (yVal < (currentLevelProperties.heightLevel_ - 1u)))
		{
			// keep track of "best" disparity for current pixel
			unsigned int bestDisparity{0u};
			U best_val{(U)bp_consts::INF_BP};
			if constexpr (DISP_VALS > 0) {
				for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
					const U val = convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValInCheckerboardPart, (yVal + 1u),
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								DISP_VALS)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValInCheckerboardPart, (yVal - 1u),
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								DISP_VALS)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								(xValInCheckerboardPart	+ checkerboardPartAdjustment), yVal,
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								DISP_VALS)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								(xValInCheckerboardPart + checkerboardPartAdjustment) - 1u, yVal,
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								DISP_VALS)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xValInCheckerboardPart, yVal,
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								DISP_VALS)]);
					if (val < best_val) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
			}
			else {
				for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
					const U val = convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValInCheckerboardPart, (yVal + 1u),
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								bpSettingsDispVals)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								xValInCheckerboardPart, (yVal - 1u),
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								bpSettingsDispVals)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								(xValInCheckerboardPart	+ checkerboardPartAdjustment), yVal,
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								bpSettingsDispVals)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessage(
								(xValInCheckerboardPart + checkerboardPartAdjustment) - 1u, yVal,
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								bpSettingsDispVals)]) +
							convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
								xValInCheckerboardPart, yVal,
								currentLevelProperties.paddedWidthCheckerboardLevel_,
								currentLevelProperties.heightLevel_,
								currentDisparity,
								bpSettingsDispVals)]);
					if (val < best_val) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
			}

			disparityBetweenImagesDevice[yVal*currentLevelProperties.widthLevel_ +
			                             (xValInCheckerboardPart * 2 + checkerboardPartAdjustment)] = bestDisparity;
		} else {
			disparityBetweenImagesDevice[yVal* currentLevelProperties.widthLevel_ +
			                             (xValInCheckerboardPart * 2 + checkerboardPartAdjustment)] = 0;
		}
	}

	//process from part 2 of checkerboard
	//adjustment based on checkerboard; need to add 1 to x for even-numbered rows for final index mapping into disparity images for checkerboard 2
	checkerboardPartAdjustment = ((yVal + 1u) % 2);

	if (withinImageBounds(xValInCheckerboardPart*2 + checkerboardPartAdjustment, yVal, currentLevelProperties.widthLevel_, currentLevelProperties.heightLevel_)) {
		if ((xValInCheckerboardPart >= (1 - checkerboardPartAdjustment)) &&
			(xValInCheckerboardPart < (currentLevelProperties.widthCheckerboardLevel_ - checkerboardPartAdjustment)) &&
			(yVal > 0) && (yVal < (currentLevelProperties.heightLevel_ - 1)))
		{
			// keep track of "best" disparity for current pixel
			unsigned int bestDisparity{0u};
			U best_val{(U)bp_consts::INF_BP};
			if constexpr (DISP_VALS > 0) {
				for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
					const U val = convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal + 1u),
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									DISP_VALS)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal - 1u),
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									DISP_VALS)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									(xValInCheckerboardPart	+ checkerboardPartAdjustment), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									DISP_VALS)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									(xValInCheckerboardPart + checkerboardPartAdjustment) - 1u,	yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									DISP_VALS)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									DISP_VALS)]);
					if (val < best_val) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
			}
			else {
				for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
					const U val = convertValToDifferentDataTypeIfNeeded<T, U>(messageUPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal + 1u),
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									bpSettingsDispVals)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(messageDPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, (yVal - 1u),
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									bpSettingsDispVals)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(messageLPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									(xValInCheckerboardPart	+ checkerboardPartAdjustment), yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									bpSettingsDispVals)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(messageRPrevStereoCheckerboard0[retrieveIndexInDataAndMessage(
									(xValInCheckerboardPart + checkerboardPartAdjustment) - 1u,	yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									bpSettingsDispVals)]) +
								convertValToDifferentDataTypeIfNeeded<T, U>(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
									xValInCheckerboardPart, yVal,
									currentLevelProperties.paddedWidthCheckerboardLevel_,
									currentLevelProperties.heightLevel_,
									currentDisparity,
									bpSettingsDispVals)]);
					if (val < best_val) {
						best_val = val;
						bestDisparity = currentDisparity;
					}
				}
			}

			disparityBetweenImagesDevice[yVal * currentLevelProperties.widthLevel_ +
			                             (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = bestDisparity;
		} else {
			disparityBetweenImagesDevice[yVal * currentLevelProperties.widthLevel_ +
			                             (xValInCheckerboardPart*2 + checkerboardPartAdjustment)] = 0;
		}
	}
}


template<typename T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsAtPointKernel(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const unsigned int bpSettingsDispVals = 0)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	}
}

template<typename T, unsigned int DISP_VALS>
ARCHITECTURE_ADDITION inline void printDataAndMessageValsToPointKernel(
		const unsigned int xVal, const unsigned int yVal, const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const unsigned int bpSettingsDispVals = 0)
{
	const unsigned int checkerboardAdjustment = (((xVal + yVal) % 2) == 0) ? ((yVal)%2) : ((yVal+1)%2);
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	}
}

#endif /* SHAREDBPPROCESSINGFUNCTS_H_ */
