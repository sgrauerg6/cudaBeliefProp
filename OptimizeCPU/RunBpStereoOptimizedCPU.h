/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include "SmoothImageCPU.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <iostream>
#include <string>

template <typename T>
class RunBpStereoOptimizedCPU : public RunBpStereoSet<T> {
public:
	RunBpStereoOptimizedCPU();
	virtual ~RunBpStereoOptimizedCPU();

	std::string getBpRunDescription() { return "Optimized CPU"; }

	//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath, const BPsettings& algSettings, std::ostream& resultsStream);
};

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
