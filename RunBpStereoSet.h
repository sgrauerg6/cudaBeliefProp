/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include "bpStereoParameters.h"
#include "SmoothImage.h"
#include "ProcessBPOnTargetDevice.h"
#include <cstring>
#include "imageHelpers.h"
#include "DetailedTimings.h"
#include "RunBpStereoSetMemoryManagement.h"

template <typename T>
class RunBpStereoSet {
public:
	RunBpStereoSet() {
	}

	virtual ~RunBpStereoSet() {
	}

	//pure abstract overloaded operator that must be defined in child class
	virtual float operator()(const char* refImagePath, const char* testImagePath,
		BPsettings algSettings, const char* saveDisparityMapImagePath, FILE* resultsFile) = 0;

protected:
	float processStereoSet(const char* refImagePath, const char* testImagePath,
		BPsettings algSettings, const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTargetDevice<T>* runBpStereo, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr);

};

#endif /* RUNBPSTEREOSET_H_ */
