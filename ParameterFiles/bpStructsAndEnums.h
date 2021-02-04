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

//structure to store the settings for the number of levels and iterations
struct BPsettings
{
	//initally set to default values
	unsigned int numLevels_{bp_params::LEVELS_BP};
	unsigned int numIterations_{bp_params::ITER_BP};
	float smoothingSigma_{bp_params::SIGMA_BP};
	float lambda_bp_{bp_params::LAMBDA_BP};
	float data_k_bp_{bp_params::DATA_K_BP};
	float disc_k_bp_{bp_params::DISC_K_BP};
};

//structure to store the properties of the current level
struct levelProperties
{
	levelProperties(const std::array<unsigned int, 2>& widthHeight = {0, 0}) :
		widthLevel_(widthHeight[0]), heightLevel_(widthHeight[1]),
		widthCheckerboardLevel_(bp_util_functs::getCheckerboardWidthTargetDevice(widthLevel_)),
		paddedWidthCheckerboardLevel_(bp_util_functs::getPaddedCheckerboardWidth(widthCheckerboardLevel_)) {}

	//get bp level properties for next (higher) level in hierarchy that processed data with half width/height of current level
	levelProperties getNextLevelProperties() const {
		return levelProperties({(unsigned int)ceil((float)widthLevel_ / 2.0f), (unsigned int)ceil((float)heightLevel_ / 2.0f)});
	}

	unsigned int widthLevel_;
	unsigned int heightLevel_;
	unsigned int widthCheckerboardLevel_;
	unsigned int paddedWidthCheckerboardLevel_;
};

//used to define the two checkerboard "parts" that the image is divided into
enum Checkerboard_Parts {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1 };
enum Message_Arrays { MESSAGES_U_CHECKERBOARD_0 = 0, MESSAGES_D_CHECKERBOARD_0, MESSAGES_L_CHECKERBOARD_0, MESSAGES_R_CHECKERBOARD_0,
	MESSAGES_U_CHECKERBOARD_1, MESSAGES_D_CHECKERBOARD_1, MESSAGES_L_CHECKERBOARD_1, MESSAGES_R_CHECKERBOARD_1 };

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
