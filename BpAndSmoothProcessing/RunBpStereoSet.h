/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "SmoothImage.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <cstring>
#include "../RuntimeTiming/DetailedTimings.h"
#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include "../OutputEvaluation/DisparityMap.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"
#include <iostream>
#include <unordered_map>
#include "../RuntimeTiming/DetailedTimings.h"
#include <memory>
#include "../ImageDataAndProcessing/BpImage.h"

struct ProcessStereoSetOutput
{
	float runTime = 0.0;
	DisparityMap<float> outDisparityMap;
};

template <typename T>
class RunBpStereoSet {
public:
	RunBpStereoSet() {
	}

	virtual ~RunBpStereoSet() {
	}

	virtual std::string getBpRunDescription() = 0;

	//pure abstract overloaded operator that must be defined in child class
	virtual ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream) = 0;

protected:
	ProcessStereoSetOutput processStereoSet(const std::string& refImagePath, const std::string& testImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream, const std::unique_ptr<SmoothImage>& smoothImage,
		const std::unique_ptr<ProcessBPOnTargetDevice<T>>& runBpStereo, const std::unique_ptr<RunBpStereoSetMemoryManagement>& runBPMemoryMangement = std::unique_ptr<RunBpStereoSetMemoryManagement>(new RunBpStereoSetMemoryManagement) );

};

#endif /* RUNBPSTEREOSET_H_ */