/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include "bpStereoCudaParameters.cuh"

class RunBpStereoSet {
public:
	RunBpStereoSet();
	virtual ~RunBpStereoSet();
	virtual float operator()(const char* refImagePath, const char* testImagePath,
			BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile) = 0;
};

#endif /* RUNBPSTEREOSET_H_ */
