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

#define TSUKUBA_IMAGES 1
#define CONES_IMAGES_QUARTER_SIZE 2
#define CONES_IMAGES_HALF_SIZE 3
#define CONES_IMAGES_FULL_SIZE 4
#define IMAGE_SET_PARAMETERS_FROM_PYTHON 5
#define IMAGE_SET_TO_PROCESS CONES_IMAGES_HALF_SIZE

namespace bp_params
{
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

	#if (IMAGE_SET_TO_PROCESS == TSUKUBA_IMAGES)

	const std::string STEREO_SET = "tsukubaSet";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	#define NUM_POSSIBLE_DISPARITY_VALUES 15

	// scaling from computed disparity to graylevel in output
	const unsigned int SCALE_BP = 16;

	//info about a default ground truth
	const std::string DEFAULT_GROUND_TRUTH_DISPARITY_FILE = "groundTruthDispTsukuba.pgm";

	#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_QUARTER_SIZE)

	const std::string STEREO_SET = "conesQuarterSize";

	#define NUM_POSSIBLE_DISPARITY_VALUES 63

	// scaling from computed disparity to graylevel in output
	const unsigned int SCALE_BP = 4;

	#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_HALF_SIZE)

	const std::string STEREO_SET = "conesHalfSize";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	#define NUM_POSSIBLE_DISPARITY_VALUES 127

	// scaling from computed disparity to graylevel in output
	const unsigned int SCALE_BP = 2;

	#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_FULL_SIZE)

	const std::string STEREO_SET = "cones";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	const unsigned int NUM_POSSIBLE_DISPARITY_VALUES = 255;

	// scaling from computed disparity to graylevel in output
	const unsigned int SCALE_BP = 1;

	//If image set parameters from python, then use settings in current iteration in python script
	//These settings are written to file bpParametersFromPython.h as part of the python script
	#elif (IMAGE_SET_TO_PROCESS == IMAGE_SET_PARAMETERS_FROM_PYTHON)

	const std::string STEREO_SET = STEREO_SET_FROM_PYTHON;

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	#define NUM_POSSIBLE_DISPARITY_VALUES NUM_POSSIBLE_DISPARITY_VALUES_FROM_PYTHON

	// scaling from computed disparity to graylevel in output
	const float SCALE_BP = SCALE_BP_FROM_PYTHON;

	// number of BP iterations at each scale/level
	const unsigned int ITER_BP = ITER_BP_FROM_PYTHON;

	// number of scales/levels in the pyramid to run BP
	const unsigned int LEVELS_BP = LEVELS_BP_FROM_PYTHON;

	//truncation of discontinuity cost
	const float DISC_K_BP = DISC_K_BP_FROM_PYTHON;

	// truncation of data cost
	const float DATA_K_BP = DATA_K_BP_FROM_PYTHON;

	// weighing of data cost
	const float LAMBDA_BP = LAMBDA_BP_FROM_PYTHON;

	// amount to smooth the input images
	const unsigned int SIGMA_BP = SIGMA_BP_FROM_PYTHON;

	#if USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT_FROM_PYTHON == 1
		#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT
	#endif

	#endif //IMAGE_SET_TO_PROCESS
}

#endif /* BPSTEREOPARAMETERS_H_ */
