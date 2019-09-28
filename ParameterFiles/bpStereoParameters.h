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

constexpr float INF_BP = 65504.0f;     // large cost (used for "infinity"), value set to support half type
const float SMALL_VAL_BP = .01f;

#define NO_EXPECTED_STEREO_BP -999.0f

enum class image_set_options
{
	TSUKUBA_IMAGES_E,
	CONES_IMAGES_QUARTER_SIZE_E,
	CONES_IMAGES_HALF_SIZE_E,
	CONES_IMAGES_FULL_SIZE_E,
	//IMAGE_SET_PARAMETERS_FROM_PYTHON_E
};

constexpr image_set_options IMAGE_SET_TO_PROCESS_E = image_set_options::TSUKUBA_IMAGES_E;

constexpr unsigned int getNumDispVals(const image_set_options imageSet)
{
	if (imageSet == image_set_options::TSUKUBA_IMAGES_E)
	{
		return 15;
	}
	if (imageSet == image_set_options::CONES_IMAGES_QUARTER_SIZE_E)
	{
		return 63;
	}
	if (imageSet == image_set_options::CONES_IMAGES_HALF_SIZE_E)
	{
		return 127;
	}
	if (imageSet == image_set_options::CONES_IMAGES_FULL_SIZE_E)
	{
		return 255;
	}

	return 0;
};

constexpr unsigned int getScaleFactor(const image_set_options imageSet)
{
	if (imageSet == image_set_options::TSUKUBA_IMAGES_E)
	{
		return 16;
	}
	if (imageSet == image_set_options::CONES_IMAGES_QUARTER_SIZE_E)
	{
		return 4;
	}
	if (imageSet == image_set_options::CONES_IMAGES_HALF_SIZE_E)
	{
		return 2;
	}
	if (imageSet == image_set_options::CONES_IMAGES_FULL_SIZE_E)
	{
		return 1;
	}

	return 0;
};

constexpr char* getStereoSetString(const image_set_options imageSet)
{
	if (imageSet == image_set_options::TSUKUBA_IMAGES_E)
	{
		return "tsukubaSet";
	}
	if (imageSet == image_set_options::CONES_IMAGES_QUARTER_SIZE_E)
	{
		return "conesQuarterSize";
	}
	if (imageSet == image_set_options::CONES_IMAGES_HALF_SIZE_E)
	{
		return "conesHalfSize";
	}
	if (imageSet == image_set_options::CONES_IMAGES_FULL_SIZE_E)
	{
		return "cones";
	}

	return nullptr;
};
constexpr unsigned int NUM_POSSIBLE_DISPARITY_VALUES = getNumDispVals(IMAGE_SET_TO_PROCESS_E);

namespace bp_params
{
	constexpr unsigned int SCALE_BP = getScaleFactor(IMAGE_SET_TO_PROCESS_E);
	constexpr char* STEREO_SET = getStereoSetString(IMAGE_SET_TO_PROCESS_E);

	//number of belief propagation stereo runs of same image set
	const unsigned int NUM_BP_STEREO_RUNS = 15;

	//define the default message value...
	const float DEFAULT_INITIAL_MESSAGE_VAL = 0.0f;

	// number of BP iterations at each scale/level
	const unsigned int ITER_BP = 10;

	// number of scales/levels in the pyramid to run BP
	const unsigned int LEVELS_BP = 5;

	//truncation of discontinuity cost
	const float DISC_K_BP = 1.7f;

	// truncation of data cost
	const float DATA_K_BP = 15.0f;

	// weighing of data cost
	const float LAMBDA_BP = 0.07f;

	// amount to smooth the input images
	const float SIGMA_BP = 0.0f;
}

#endif /* BPSTEREOPARAMETERS_H_ */
