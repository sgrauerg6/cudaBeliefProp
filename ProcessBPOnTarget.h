/*
 * ProcessBPOnTarget.h
 *
 *  Created on: Jun 19, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGET_H_
#define PROCESSBPONTARGET_H_

#include <stdlib.h>
#include "DetailedTimings.h"
#include "bpStereoParameters.h"

template<typename T>
class ProcessBPOnTarget {
public:
	ProcessBPOnTarget() { };
	virtual ~ProcessBPOnTarget() { };

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
	//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
	virtual DetailedTimings* operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings) = 0;
};

#endif /* PROCESSBPONTARGET_H_ */
