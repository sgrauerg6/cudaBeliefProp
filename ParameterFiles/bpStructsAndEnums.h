/*
 * bpStructsAndEnums.h
 *
 *  Created on: Sep 22, 2019
 *      Author: scott
 */

#ifndef BPSTRUCTSANDENUMS_H_
#define BPSTRUCTSANDENUMS_H_

#include <array>
#include "bpStereoParameters.h"
#include "BpAndSmoothProcessing/BpUtilFuncts.h"
#include "../BpAndSmoothProcessing/BpUtilFuncts.h"

//structure to store the settings for the number of levels and iterations
struct BPsettings
{
	//initally set to default values
	unsigned int numLevels_{bp_params::LEVELS_BP};
	unsigned int numIterations_{bp_params::ITER_BP};
	float smoothingSigma_{bp_params::SIGMA_BP};
	float lambda_bp_{bp_params::LAMBDA_BP};
	float data_k_bp_{bp_params::DATA_K_BP};
	float disc_k_bp_{bp_params::DISC_K_BP[0]};
	unsigned int numDispVals_{0};
};

//structure to store the properties of the current level
struct levelProperties
{
	levelProperties(const std::array<unsigned int, 2>& widthHeight = {0, 0}, unsigned long offsetIntoArrays = 0) :
		widthLevel_(widthHeight[0]), heightLevel_(widthHeight[1]),
		widthCheckerboardLevel_(getCheckerboardWidthTargetDevice(widthLevel_)),
		paddedWidthCheckerboardLevel_(getPaddedCheckerboardWidth(widthCheckerboardLevel_)),
		offsetIntoArrays_(offsetIntoArrays) {}

	//get bp level properties for next (higher) level in hierarchy that processed data with half width/height of current level
	template <typename T>
	levelProperties getNextLevelProperties(const unsigned int numDisparityValues) const {
		const auto offsetNextLevel = offsetIntoArrays_ + getNumDataInBpArrays<T>(numDisparityValues);
		return levelProperties({(unsigned int)ceil((float)widthLevel_ / 2.0f), (unsigned int)ceil((float)heightLevel_ / 2.0f)},
				offsetNextLevel);
	}

	//get the amount of data in each BP array (data cost/messages for each checkerboard) at the current level
	//with the given number of possible movements
	template <typename T>
	unsigned int getNumDataInBpArrays(const unsigned int numDisparityValues) const {
		return getNumDataForAlignedMemoryAtLevel<T>({widthLevel_, heightLevel_}, numDisparityValues);
	}

	unsigned int getCheckerboardWidthTargetDevice(const unsigned int widthLevel) const {
		return (unsigned int)std::ceil(((float)widthLevel) / 2.0f);
	}

	unsigned int getPaddedCheckerboardWidth(const unsigned int checkerboardWidth) const
	{
		//add "padding" to checkerboard width if necessary for alignment
		return ((checkerboardWidth % bp_params::NUM_DATA_ALIGN_WIDTH) == 0) ?
				checkerboardWidth :
				(checkerboardWidth + (bp_params::NUM_DATA_ALIGN_WIDTH - (checkerboardWidth % bp_params::NUM_DATA_ALIGN_WIDTH)));
	}

	template <typename T>
	unsigned long getNumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& widthHeightLevel,
			const unsigned int totalPossibleMovements) const
	{
		const unsigned long numDataAtLevel = (unsigned long)getPaddedCheckerboardWidth(getCheckerboardWidthTargetDevice(widthHeightLevel[0]))
			* ((unsigned long)widthHeightLevel[1]) * (unsigned long)totalPossibleMovements;
		unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

		if ((numBytesAtLevel % bp_params::BYTES_ALIGN_MEMORY) == 0) {
			return numDataAtLevel;
		}
		else {
			numBytesAtLevel += (bp_params::BYTES_ALIGN_MEMORY - (numBytesAtLevel % bp_params::BYTES_ALIGN_MEMORY));
			return (numBytesAtLevel / sizeof(T));
		}
	}

	template <typename T>
	static unsigned long getTotalDataForAlignedMemoryAllLevels(const std::array<unsigned int, 2>& widthHeightBottomLevel,
			const unsigned int totalPossibleMovements, const unsigned int numLevels)
	{
		levelProperties currLevelProperties(widthHeightBottomLevel);
		unsigned long totalData = currLevelProperties.getNumDataInBpArrays<T>(totalPossibleMovements);
		for (unsigned int currLevelNum = 1; currLevelNum < numLevels; currLevelNum++) {
			currLevelProperties = currLevelProperties.getNextLevelProperties<T>(totalPossibleMovements);
			totalData += currLevelProperties.getNumDataInBpArrays<T>(totalPossibleMovements);
		}

		return totalData;
	}


	unsigned int widthLevel_;
	unsigned int heightLevel_;
	unsigned int widthCheckerboardLevel_;
	unsigned int paddedWidthCheckerboardLevel_;
	unsigned long offsetIntoArrays_;
};

//used to define the two checkerboard "parts" that the image is divided into
enum Checkerboard_Parts {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1 };
enum Message_Arrays { MESSAGES_U_CHECKERBOARD_0 = 0, MESSAGES_D_CHECKERBOARD_0, MESSAGES_L_CHECKERBOARD_0, MESSAGES_R_CHECKERBOARD_0,
	MESSAGES_U_CHECKERBOARD_1, MESSAGES_D_CHECKERBOARD_1, MESSAGES_L_CHECKERBOARD_1, MESSAGES_R_CHECKERBOARD_1 };
enum class messageComp { U_MESSAGE, D_MESSAGE, L_MESSAGE, R_MESSAGE };

template <class T>
struct checkerboardMessages
{
	//each checkerboard messages element corresponds to separate Message_Arrays enum that go from 0 to 7 (8 total)
	//could use a map/unordered map to map Message_Arrays enum to corresponding message array but using array structure is likely faster
	std::array<T, 8> checkerboardMessagesAtLevel_;
};

template <class T>
struct dataCostData
{
	T dataCostCheckerboard0_;
	T dataCostCheckerboard1_;
};

#endif /* BPSTRUCTSANDENUMS_H_ */
