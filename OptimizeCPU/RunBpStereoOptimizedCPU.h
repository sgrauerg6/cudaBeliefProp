/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include "SmoothImageCPU.h"
#include "RunBpStereoSet.h"
#include "ProcessBPOnTargetDevice.h"
#include <iostream>

template <typename T>
class RunBpStereoOptimizedCPU : public RunBpStereoSet<T> {
public:
	RunBpStereoOptimizedCPU();
	virtual ~RunBpStereoOptimizedCPU();

	//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
	ProcessStereoSetOutput operator()(const char* refImagePath, const char* testImagePath, const BPsettings& algSettings, std::ostream& resultsFile);
};

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
