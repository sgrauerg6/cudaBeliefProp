/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include "SmoothImageCPU.h"
#include "imageHelpers.h"
#include "BpStereoProcessingOptimizedCPU.h"

class RunBpStereoOptimizedCPU {
public:
	RunBpStereoOptimizedCPU();
	virtual ~RunBpStereoOptimizedCPU();

	//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
	float operator()(const char* refImagePath, const char* testImagePath, BPsettings algSettings, const char* saveDisparityMapImagePath, FILE* resultsFile);
};

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
