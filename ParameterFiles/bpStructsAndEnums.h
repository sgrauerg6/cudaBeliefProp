/*
 * bpStructsAndEnums.h
 *
 *  Created on: Sep 22, 2019
 *      Author: scott
 */

#ifndef BPSTRUCTSANDENUMS_H_
#define BPSTRUCTSANDENUMS_H_

#include "bpStereoParameters.h"

//structure to store the settings for the number of levels and iterations
struct BPsettings
{
	int numLevels;
	int numIterations;

	float smoothingSigma;
	float lambda_bp;
	float data_k_bp;
	float disc_k_bp;

	//default constructor setting BPsettings
	//to default values
	BPsettings()
	{
		smoothingSigma = bp_params::SIGMA_BP;
		numLevels = bp_params::LEVELS_BP;
		numIterations = bp_params::ITER_BP;
		lambda_bp = bp_params::LAMBDA_BP;
		data_k_bp = bp_params::DATA_K_BP;
		disc_k_bp = bp_params::DISC_K_BP;
	}
};

//structure to store the properties of the current level
struct levelProperties
{
	int widthLevel;
	int widthCheckerboardLevel;
	int paddedWidthCheckerboardLevel;
	int heightLevel;
};

//used to define the two checkerboard "parts" that the image is divided into
enum Checkerboard_Parts {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1 };

template <class T>
struct checkerboardMessages
{
	T messagesU;
	T messagesD;
	T messagesL;
	T messagesR;
};

template <class T>
struct dataCostData
{
	T dataCostCheckerboard0;
	T dataCostCheckerboard1;
};

#endif /* BPSTRUCTSANDENUMS_H_ */
