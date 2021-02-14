/*
 * bpStereoParameters.h
 *
 *  Created on: Jun 18, 2019
 *      Author: scott
 */

/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//This class defines parameters for the cuda implementation for disparity map estimation for a pair of stereo images

#ifndef BPSTEREOPARAMETERS_H_
#define BPSTEREOPARAMETERS_H_

#include "bpParametersFromPython.h"
#include <string>
#include <array>

namespace bp_consts
{
	constexpr float INF_BP = 65504.0f;     // large cost (used for "infinity"), value set to support half type
}

enum image_set_options
{
	TSUKUBA_IMAGES_E = 0,
	VENUS_IMAGES_E = 1,
	BARN_1_IMAGES_E = 2,
	CONES_IMAGES_QUARTER_SIZE_E = 3,
	CONES_IMAGES_HALF_SIZE_E = 4,
	//CONES_IMAGES_FULL_SIZE_E,
	//IMAGE_SET_PARAMETERS_FROM_PYTHON_E
};

constexpr image_set_options IMAGE_SET_TO_PROCESS_E{image_set_options::CONES_IMAGES_HALF_SIZE_E};

struct BpStereoSet {
	const char* name;
	unsigned int numDispVals;
	unsigned int scaleFactor;
};

constexpr std::array<BpStereoSet, 5> STEREO_SETS_TO_PROCESS{
	//declare stereo sets to process with (in order) name, num disparity values, and scale factor
	//order is the same as in image_set_options enum
	BpStereoSet{"tsukubaSet", 16, 16},
	BpStereoSet{"venus", 21, 8},
	BpStereoSet{"barn1", 32, 8},
	BpStereoSet{"conesQuarterSize", 64, 4},
	BpStereoSet{"conesHalfSize", 128, 2}
};

namespace bp_params
{
	constexpr std::array<unsigned int, 5> NUM_POSSIBLE_DISPARITY_VALUES{
		STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].numDispVals,
		STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].numDispVals,
		STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].numDispVals,
		STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].numDispVals,
		STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].numDispVals};
	constexpr std::array<unsigned int, 5> SCALE_BP{
		STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].scaleFactor,
		STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].scaleFactor,
		STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].scaleFactor,
		STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].scaleFactor,
		STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].scaleFactor};
	constexpr std::array<const char*, 5> STEREO_SET{
		STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].name,
		STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].name,
		STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].name,
		STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].name,
		STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].name};

	//number of belief propagation stereo runs of same image set
	constexpr unsigned int NUM_BP_STEREO_RUNS = 15;

	//define the default message value...
	constexpr float DEFAULT_INITIAL_MESSAGE_VAL = 0.0f;

	// number of BP iterations at each scale/level
	constexpr unsigned int ITER_BP = 7;

	// number of scales/levels in the pyramid to run BP
	constexpr unsigned int LEVELS_BP = 5;

	//truncation of discontinuity cost
	constexpr std::array<float, 5> DISC_K_BP{
		(float)STEREO_SETS_TO_PROCESS[TSUKUBA_IMAGES_E].numDispVals / 7.5f,
		(float)STEREO_SETS_TO_PROCESS[VENUS_IMAGES_E].numDispVals / 7.5f,
		(float)STEREO_SETS_TO_PROCESS[BARN_1_IMAGES_E].numDispVals / 7.5f,
		(float)STEREO_SETS_TO_PROCESS[CONES_IMAGES_QUARTER_SIZE_E].numDispVals / 7.5f,
		(float)STEREO_SETS_TO_PROCESS[CONES_IMAGES_HALF_SIZE_E].numDispVals / 7.5f};

	// truncation of data cost
	constexpr float DATA_K_BP = 15.0f;

	// weighing of data cost
	constexpr float LAMBDA_BP = 0.1f;

	// amount to smooth the input images
	constexpr float SIGMA_BP = 0.0f;
}

#endif /* BPSTEREOPARAMETERS_H_ */
