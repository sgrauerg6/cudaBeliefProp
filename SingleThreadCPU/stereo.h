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
#include <array>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "filter.h"
#include "imconv.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../ParameterFiles/bpRunSettings.h"

template<typename T, unsigned int DISP_VALS>
class RunBpStereoCPUSingleThread : public RunBpStereoSet<T, DISP_VALS>
{
public:
	ProcessStereoSetOutput operator()(const std::array<std::string, 2>& refTestImagePath, const BPsettings& algSettings, std::ostream& resultsFile) override;
	std::string getBpRunDescription() override { return "Single-Thread CPU"; }

private:
	// compute message
	image<float[DISP_VALS]> *comp_data(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings);
	void msg(float s1[DISP_VALS], float s2[DISP_VALS], float s3[DISP_VALS], float s4[DISP_VALS],
			float dst[DISP_VALS], const float disc_k_bp);
	image<uchar> *output(image<float[DISP_VALS]> *u, image<float[DISP_VALS]> *d,
			image<float[DISP_VALS]> *l, image<float[DISP_VALS]> *r,
			image<float[DISP_VALS]> *data);
	void bp_cb(image<float[DISP_VALS]> *u, image<float[DISP_VALS]> *d,
			image<float[DISP_VALS]> *l, image<float[DISP_VALS]> *r,
			image<float[DISP_VALS]> *data, const unsigned int iter, const float disc_k_bp);
	image<uchar> *stereo_ms(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings, std::ostream& resultsFile, float& runtime);
};

#ifdef _WIN32

extern "C" __declspec(dllexport) RunBpStereoSet<float>* __cdecl createRunBpStereoCPUSingleThreadFloat();
extern "C" __declspec(dllexport) RunBpStereoSet<double>* __cdecl createRunBpStereoCPUSingleThreadDouble();
extern "C" __declspec(dllexport) RunBpStereoSet<short>* __cdecl createRunBpStereoCPUSingleThreadShort();

#endif //_WIN32

#endif /* STEREO_H_ */
