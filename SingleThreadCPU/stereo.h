/*
 * stereo.h
 *
 *  Created on: Feb 4, 2017
 *      Author: scottgg
 */

#ifndef STEREO_H_
#define STEREO_H_

#include "RunBpStereoSet.h"
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
#include "bpStereoCudaParameters.h"
#include <chrono>

#define MAX_ALLOWED_LEVELS 10
#define INF 1E20     // large cost
#define VALUES NUM_POSSIBLE_DISPARITY_VALUES    // number of possible disparities
#define SCALE SCALE_BP     // scaling from disparity to graylevel in output

class RunBpStereoCPUSingleThread : public RunBpStereoSet
{
public:
	float operator()(const char* refImagePath, const char* testImagePath, BPsettings algSettings, const char* saveDisparityImagePath, FILE* resultsFile);

private:
	// compute message
	image<float[VALUES]> *comp_data(image<uchar> *img1, image<uchar> *img2, BPsettings algSettings);
	void msg(float s1[VALUES], float s2[VALUES], float s3[VALUES], float s4[VALUES],
			float dst[VALUES], float disc_k_bp);
	image<uchar> *output(image<float[VALUES]> *u, image<float[VALUES]> *d,
			image<float[VALUES]> *l, image<float[VALUES]> *r,
			image<float[VALUES]> *data);
	void bp_cb(image<float[VALUES]> *u, image<float[VALUES]> *d,
			image<float[VALUES]> *l, image<float[VALUES]> *r,
			image<float[VALUES]> *data, int iter, float disc_k_bp);
	image<uchar> *stereo_ms(image<uchar> *img1, image<uchar> *img2, BPsettings algSettings, FILE* resultsFile, float& runtime);
};
#endif /* STEREO_H_ */
