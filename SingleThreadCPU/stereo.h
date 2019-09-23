/*
 * stereo.h
 *
 *  Created on: Feb 4, 2017
 *      Author: scottgg
 */

#ifndef STEREO_H_
#define STEREO_H_

#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "filter.h"
#include "imconv.h"
#include <string>
#include "ParameterFiles/bpStereoParameters.h"
#include "ParameterFiles/bpStructsAndEnums.h"
#include <chrono>
#include "../BpAndSmoothProcessing/SmoothImage.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <iostream>

template<typename T>
class RunBpStereoCPUSingleThread : public RunBpStereoSet<T>
{
public:
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath, const BPsettings& algSettings, std::ostream& resultsFile);
	std::string getBpRunDescription() { return "Single-Thread CPU"; }

private:
	// compute message
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *comp_data(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings);
	void msg(float s1[NUM_POSSIBLE_DISPARITY_VALUES], float s2[NUM_POSSIBLE_DISPARITY_VALUES], float s3[NUM_POSSIBLE_DISPARITY_VALUES], float s4[NUM_POSSIBLE_DISPARITY_VALUES],
			float dst[NUM_POSSIBLE_DISPARITY_VALUES], float disc_k_bp);
	image<uchar> *output(image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *u, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *d,
			image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *l, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *r,
			image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *data);
	void bp_cb(image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *u, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *d,
			image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *l, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *r,
			image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *data, int iter, float disc_k_bp);
	image<uchar> *stereo_ms(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings, std::ostream& resultsFile, float& runtime);
};
#endif /* STEREO_H_ */
