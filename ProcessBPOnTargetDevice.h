/*
 * ProcessBPOnTargetDevice.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICE_H_
#define PROCESSBPONTARGETDEVICE_H_

#include "DetailedTimings.h"
#include "ProcessBPOnTargetDeviceHelperFuncts.h"
#include "bpStereoParameters.h"
#include <math.h>

template<typename T>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice();
	virtual ~ProcessBPOnTargetDevice();

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
	virtual DetailedTimings* operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings, ProcessBPOnTargetDeviceHelperFuncts<beliefPropProcessingDataType>* processBPHelperFuncts = nullptr);
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
