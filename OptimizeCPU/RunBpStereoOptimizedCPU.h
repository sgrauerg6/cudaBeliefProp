/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include <iostream>
#include <string>
#include <memory>
#include "SmoothImageCPU.h"
#include "ProcessOptimizedCPUBP.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"

template <typename T = float>
class RunBpStereoOptimizedCPU : public RunBpStereoSet<T> {
public:
	RunBpStereoOptimizedCPU();
	virtual ~RunBpStereoOptimizedCPU();

	std::string getBpRunDescription() override { return "Optimized CPU"; }

	//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath, const BPsettings& algSettings, std::ostream& resultsStream) override;
};

extern "C" __declspec(dllexport) RunBpStereoSet<float>* __cdecl createRunBpStereoOptimizedCPUFloat();
extern "C" __declspec(dllexport) RunBpStereoSet<double>* __cdecl createRunBpStereoOptimizedCPUDouble();
extern "C" __declspec(dllexport) RunBpStereoSet<short>* __cdecl createRunBpStereoOptimizedCPUShort();

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
