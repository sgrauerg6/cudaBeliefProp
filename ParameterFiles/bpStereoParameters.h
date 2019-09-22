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

#include <stdio.h>
#include "bpParametersFromPython.h"
#include <vector>
#include <algorithm>
#include <string>

const float INF_BP = 65504.0f;     // large cost (used for "infinity"), value set to support half type
const float SMALL_VAL_BP = .01f;

namespace bp_params
{
	//number of belief propagation stereo runs of same image set
	const unsigned int NUM_BP_STEREO_RUNS = 15;

	//define the default message value...
	const float DEFAULT_INITIAL_MESSAGE_VAL = 0.0f;

	#define NO_EXPECTED_STEREO_BP -999.0f

	#define TSUKUBA_IMAGES 1
	#define CONES_IMAGES_QUARTER_SIZE 2
	#define CONES_IMAGES_HALF_SIZE 3
	#define CONES_IMAGES_FULL_SIZE 4
	#define IMAGE_SET_PARAMETERS_FROM_PYTHON 5
	#define IMAGE_SET_TO_PROCESS TSUKUBA_IMAGES

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

	//define the path for the 'default' reference and test images and the output "movement" images (can easily run
	//on other images using runBpStereoImageSeries on any number of images)
	const std::string DEFAULT_REF_IMAGE_PATH = "refImageTsukuba.pgm";
	const std::string DEFAULT_TEST_IMAGE_PATH = "testImageTsukuba.pgm";

	const std::string SAVE_DISPARITY_IMAGE_PATH_1 = "computedDisparityMapTsukuba1.pgm";
	const std::string SAVE_DISPARITY_IMAGE_PATH_2 = "computedDisparityMapTsukuba2.pgm";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	#define NUM_POSSIBLE_DISPARITY_VALUES 15

	// scaling from computed disparity to graylevel in output
	const unsigned int SCALE_BP = 16;

	//info about a default ground truth
	const std::string DEFAULT_GROUND_TRUTH_DISPARITY_FILE = "groundTruthDispTsukuba.pgm";

	#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_QUARTER_SIZE)

	//define the path for the 'default' reference and test images and the output "movement" images (can easily run
	//on other images using runBpStereoImageSeries on any number of images)
	const std::string DEFAULT_REF_IMAGE_PATH = "conesQuarter2.pgm";
	const std::string DEFAULT_TEST_IMAGE_PATH = "conesQuarter6.pgm";

	const std::string SAVE_DISPARITY_IMAGE_PATH_1 = "computedDisparityConesQuarter1.pgm";
	const std::string SAVE_DISPARITY_IMAGE_PATH_2 = "computedDisparityConesQuarter2.pgm";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	const unsigned int NUM_POSSIBLE_DISPARITY_VALUES = 63;

	// scaling from computed disparity to graylevel in output
	const float SCALE_BP = 4.0f;

	//info about a default ground truth
	const std::string DEFAULT_GROUND_TRUTH_DISPARITY_FILE = "conesQuarterGroundTruth.pgm";

	#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_HALF_SIZE)

	//define the path for the 'default' reference and test images and the output "movement" images (can easily run
	//on other images using runBpStereoImageSeries on any number of images)
	const std::string DEFAULT_REF_IMAGE_PATH = "conesHalf2.pgm";
	const std::string DEFAULT_TEST_IMAGE_PATH = "conesHalf6.pgm";

	const std::string SAVE_DISPARITY_IMAGE_PATH_1 = "computedDisparityConesHalf1.pgm";
	const std::string SAVE_DISPARITY_IMAGE_PATH_2 = "computedDisparityConesHalf2.pgm";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	const unsigned int NUM_POSSIBLE_DISPARITY_VALUES = 127;

	// scaling from computed disparity to graylevel in output
	const float SCALE_BP = 2.0f;

	//info about a default ground truth
	const std::string DEFAULT_GROUND_TRUTH_DISPARITY_FILE = "conesHalfGroundTruth.pgm";

	#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_FULL_SIZE)

	//define the path for the 'default' reference and test images and the output "movement" images (can easily run
	//on other images using runBpStereoImageSeries on any number of images)
	const std::string DEFAULT_REF_IMAGE_PATH = "conesFull2.pgm";
	const std::string DEFAULT_TEST_IMAGE_PATH = "conesFull2.pgm";

	const std::string SAVE_DISPARITY_IMAGE_PATH_1 = "computedDisparityConesFull1.pgm";
	const std::string SAVE_DISPARITY_IMAGE_PATH_2 = "computedDisparityConesFull2.pgm";

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	const unsigned int NUM_POSSIBLE_DISPARITY_VALUES = 255;

	// scaling from computed disparity to graylevel in output
	const float SCALE_BP = 1.0f;

	//info about a default ground truth
	const std::string DEFAULT_GROUND_TRUTH_DISPARITY_FILE = "conesFullGroundTruth.pgm";

	//If image set parameters from python, then use settings in current iteration in python script
	//These settings are written to file bpParametersFromPython.h as part of the python script
	#elif (IMAGE_SET_TO_PROCESS == IMAGE_SET_PARAMETERS_FROM_PYTHON)

	//define the path for the 'default' reference and test images and the output "movement" images (can easily run
	//on other images using runBpStereoImageSeries on any number of images)
	const std::string DEFAULT_REF_IMAGE_PATH = REF_IMAGE_FROM_PYTHON;
	const std::string DEFAULT_TEST_IMAGE_PATH = TEST_IMAGE_FROM_PYTHON;

	const std::string SAVE_DISPARITY_IMAGE_PATH_1 = SAVE_DISPARITY_IMAGE_PATH_GPU_FROM_PYTHON;
	const std::string SAVE_DISPARITY_IMAGE_PATH_2 = SAVE_DISPARITY_IMAGE_PATH_CPU_FROM_PYTHON;

	//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
	const unsigned int NUM_POSSIBLE_DISPARITY_VALUES = NUM_POSSIBLE_DISPARITY_VALUES_FROM_PYTHON;

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
	const float SIGMA_BP = SIGMA_BP_FROM_PYTHON;

	//info about a default ground truth
	const std::string DEFAULT_GROUND_TRUTH_DISPARITY_FILE = DEFAULT_GROUND_TRUTH_DISPARITY_FILE_FROM_PYTHON;

	//scaling from ground truth disparity to ground truth disparity map image
	const float DEFAULT_SCALE_GROUND_TRUTH_DISPARITY = DEFAULT_GROUND_TRUTH_DISPARITY_SCALE_FROM_PYTHON;

	#if USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT_FROM_PYTHON == 1
		#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT
	#endif

	#endif //IMAGE_SET_TO_PROCESS
}

//structure to store the settings for the number of levels and iterations
struct BPsettings
{
	int numLevels;
	int numIterations;

	float smoothingSigma;
	float lambda_bp;
	float data_k_bp;
	float disc_k_bp;

	//default constructor setting BPsettings
	//to default values
	BPsettings()
	{
		smoothingSigma = bp_params::SIGMA_BP;
		numLevels = bp_params::LEVELS_BP;
		numIterations = bp_params::ITER_BP;
		lambda_bp = bp_params::LAMBDA_BP;
		data_k_bp = bp_params::DATA_K_BP;
		disc_k_bp = bp_params::DISC_K_BP;
	}
};

//structure to store the properties of the current level
struct levelProperties
{
	int widthLevel;
	int widthCheckerboardLevel;
	int paddedWidthCheckerboardLevel;
	int heightLevel;
};

//used to define the two checkerboard "parts" that the image is divided into
enum Checkerboard_Parts {CHECKERBOARD_PART_1, CHECKERBOARD_PART_2 };

#endif /* BPSTEREOPARAMETERS_H_ */
