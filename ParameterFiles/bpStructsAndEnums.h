/*
 * bpStructsAndEnums.h
 *
 *  Created on: Sep 22, 2019
 *      Author: scott
 */

#ifndef BPSTRUCTSANDENUMS_H_
#define BPSTRUCTSANDENUMS_H_

#include "bpStereoParameters.h"
#include <array>

//structure to store the settings for the number of levels and iterations
struct BPsettings
{
	//initally set to default values
	unsigned int numLevels{bp_params::LEVELS_BP};
	unsigned int numIterations{bp_params::ITER_BP};
	float smoothingSigma{bp_params::SIGMA_BP};
	float lambda_bp{bp_params::LAMBDA_BP};
	float data_k_bp{bp_params::DATA_K_BP};
	float disc_k_bp{bp_params::DISC_K_BP};
};

//structure to store the properties of the current level
struct levelProperties
{
	unsigned int widthLevel;
	unsigned int widthCheckerboardLevel;
	unsigned int paddedWidthCheckerboardLevel;
	unsigned int heightLevel;
};

//used to define the two checkerboard "parts" that the image is divided into
enum Checkerboard_Parts {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1 };
enum Message_Arrays { MESSAGES_U_CHECKERBOARD_0 = 0, MESSAGES_D_CHECKERBOARD_0, MESSAGES_L_CHECKERBOARD_0, MESSAGES_R_CHECKERBOARD_0,
	MESSAGES_U_CHECKERBOARD_1, MESSAGES_D_CHECKERBOARD_1, MESSAGES_L_CHECKERBOARD_1, MESSAGES_R_CHECKERBOARD_1 };

template <class T>
struct checkerboardMessages
{
	std::array<T, 8> checkerboardMessagesAtLevel;
};

template <class T>
struct dataCostData
{
	T dataCostCheckerboard0;
	T dataCostCheckerboard1;
};

#endif /* BPSTRUCTSANDENUMS_H_ */
