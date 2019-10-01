/*
 * stereo.h
 *
 *  Created on: Feb 4, 2017
 *      Author: scottgg
 */

#ifndef STEREO_H_
#define STEREO_H_

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <string>
#include <chrono>
#include <iostream>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "filter.h"
#include "imconv.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../ParameterFiles/bpRunSettings.h"

template<typename T = float>
class RunBpStereoCPUSingleThread : public RunBpStereoSet<T>
{
public:
	ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath, const BPsettings& algSettings, std::ostream& resultsFile) override;
	std::string getBpRunDescription() override { return "Single-Thread CPU"; }

private:
	// compute message
	image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *comp_data(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings);
	void msg(float s1[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float s2[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float s3[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float s4[bp_params::NUM_POSSIBLE_DISPARITY_VALUES],
			float dst[bp_params::NUM_POSSIBLE_DISPARITY_VALUES], float disc_k_bp);
	image<uchar> *output(image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *u, image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *d,
			image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *l, image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *r,
			image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *data);
	void bp_cb(image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *u, image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *d,
			image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *l, image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *r,
			image<float[bp_params::NUM_POSSIBLE_DISPARITY_VALUES]> *data, int iter, float disc_k_bp);
	image<uchar> *stereo_ms(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings, std::ostream& resultsFile, float& runtime);
};

#if _WIN32

extern "C" __declspec(dllexport) RunBpStereoSet<float>* __cdecl createRunBpStereoCPUSingleThreadFloat();
extern "C" __declspec(dllexport) RunBpStereoSet<double>* __cdecl createRunBpStereoCPUSingleThreadDouble();
extern "C" __declspec(dllexport) RunBpStereoSet<short>* __cdecl createRunBpStereoCPUSingleThreadShort();

#endif //_WIN32

#endif /* STEREO_H_ */
