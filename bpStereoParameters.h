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

#define TSUKUBA_IMAGES 1
#define CONES_IMAGES_QUARTER_SIZE 2
#define CONES_IMAGES_HALF_SIZE 3
#define CONES_IMAGES_FULL_SIZE 4
#define IMAGE_SET_PARAMETERS_FROM_PYTHON 5
#define IMAGE_SET_TO_PROCESS TSUKUBA_IMAGES

#if (IMAGE_SET_TO_PROCESS == TSUKUBA_IMAGES)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "refImageTsukuba.pgm"
#define DEFAULT_TEST_IMAGE_PATH "testImageTsukuba.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_1 "computedDisparityMapTsukuba1.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_2 "computedDisparityMapTsukuba2.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 15

#define SCALE_BP 16.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "groundTruthDispTsukuba.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 16.0f //scaling from ground truth disparity to ground truth disparity map image

// number of BP iterations at each scale/level
#define ITER_BP 10

// number of scales/levels in the pyramid to run BP
#define LEVELS_BP 5

//truncation of discontinuity cost
#define DISC_K_BP 1.7f

// truncation of data cost
#define DATA_K_BP 15.0f

// weighing of data cost
#define LAMBDA_BP 0.07f

#define SIGMA_BP 0.0f    // amount to smooth the input images

#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_QUARTER_SIZE)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "conesQuarter2.pgm"
#define DEFAULT_TEST_IMAGE_PATH "conesQuarter6.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_1 "computedDisparityConesQuarter1.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_2 "computedDisparityConesQuarter2.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 63

#define SCALE_BP 4.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "conesQuarterGroundTruth.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 4.0f //scaling from ground truth disparity to ground truth disparity map image

// number of BP iterations at each scale/level
#define ITER_BP 10

// number of scales/levels in the pyramid to run BP
#define LEVELS_BP 6

//truncation of discontinuity cost
#define DISC_K_BP 1.7f

// truncation of data cost
#define DATA_K_BP 15.0f

// weighing of data cost
#define LAMBDA_BP 0.07f

#define SIGMA_BP 0.7f    // amount to smooth the input images

#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_HALF_SIZE)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "conesHalf2.pgm"
#define DEFAULT_TEST_IMAGE_PATH "conesHalf6.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_1 "computedDisparityConesHalf1.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_2 "computedDisparityConesHalf2.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 90

#define SCALE_BP 2.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "conesHalfGroundTruth.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 2.0f //scaling from ground truth disparity to ground truth disparity map image

// number of BP iterations at each scale/level
#define ITER_BP 10

// number of scales/levels in the pyramid to run BP
#define LEVELS_BP 7

//truncation of discontinuity cost
#define DISC_K_BP 1.7f

// truncation of data cost
#define DATA_K_BP 15.0f

// weighing of data cost
#define LAMBDA_BP 0.07f

#define SIGMA_BP 0.7f    // amount to smooth the input images

#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_FULL_SIZE)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "conesFull2.pgm"
#define DEFAULT_TEST_IMAGE_PATH "conesFull6.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_1 "computedDisparityConesFull1.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_2 "computedDisparityConesFull2.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 255

#define SCALE_BP 1.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "conesFullGroundTruth.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 1.0f //scaling from ground truth disparity to ground truth disparity map image

// number of BP iterations at each scale/level
#define ITER_BP 8

// number of scales/levels in the pyramid to run BP
#define LEVELS_BP 10

//truncation of discontinuity cost
#define DISC_K_BP 1.7f

// truncation of data cost
#define DATA_K_BP 15.0f

// weighing of data cost
#define LAMBDA_BP 0.07f

#define SIGMA_BP 0.7f    // amount to smooth the input images


//If image set parameters from python, then use settings in current iteration in python script
//These settings are written to file bpParametersFromPython.h as part of the python script
#elif (IMAGE_SET_TO_PROCESS == IMAGE_SET_PARAMETERS_FROM_PYTHON)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH REF_IMAGE_FROM_PYTHON
#define DEFAULT_TEST_IMAGE_PATH TEST_IMAGE_FROM_PYTHON

#define SAVE_DISPARITY_IMAGE_PATH_1 SAVE_DISPARITY_IMAGE_PATH_GPU_FROM_PYTHON
#define SAVE_DISPARITY_IMAGE_PATH_2 SAVE_DISPARITY_IMAGE_PATH_CPU_FROM_PYTHON

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES NUM_POSSIBLE_DISPARITY_VALUES_FROM_PYTHON

#define SCALE_BP SCALE_BP_FROM_PYTHON     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE DEFAULT_GROUND_TRUTH_DISPARITY_FILE_FROM_PYTHON
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY DEFAULT_GROUND_TRUTH_DISPARITY_SCALE_FROM_PYTHON //scaling from ground truth disparity to ground truth disparity map image

// number of BP iterations at each scale/level
#define ITER_BP ITER_BP_FROM_PYTHON

// number of scales/levels in the pyramid to run BP
#define LEVELS_BP LEVELS_BP_FROM_PYTHON

//truncation of discontinuity cost
#define DISC_K_BP DISC_K_BP_FROM_PYTHON

// truncation of data cost
#define DATA_K_BP DATA_K_BP_FROM_PYTHON

// weighing of data cost
#define LAMBDA_BP LAMBDA_BP_FROM_PYTHON

#define SIGMA_BP SIGMA_BP_FROM_PYTHON    // amount to smooth the input images

#if USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT_FROM_PYTHON == 1
	#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT
#endif

#endif //IMAGE_SET_TO_PROCESS

#define DATA_TYPE_PROCESSING_FLOAT 0
#define DATA_TYPE_PROCESSING_DOUBLE 1
#define DATA_TYPE_PROCESSING_HALF 2
#define DATA_TYPE_PROCESSING_HALF_TWO 3

#define USE_DEFAULT 0
#define USE_AVX_256 1
#define USE_AVX_512 2

//If image set parameters from python, then use optimization settings set in current iteration in python script
//These settings are written to file bpParametersFromPython.h as part of the python script
#if (IMAGE_SET_TO_PROCESS == IMAGE_SET_PARAMETERS_FROM_PYTHON)
#define CURRENT_DATA_TYPE_PROCESSING CURRENT_DATA_TYPE_PROCESSING_FROM_PYTHON
#define OPTIMIZED_INDEXING_SETTING OPTIMIZED_INDEXING_SETTING_FROM_PYTHON
#define CPU_OPTIMIZATION_SETTING CPU_OPTIMIZATION_SETTING_FROM_PYTHON
#else
//by default, 32-bit float data is used with optimized GPU memory management and optimized indexing
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work) by commenting out the "#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT" line
//May be able to speed up processing by switching to using 16-bit half data by setting CURRENT_DATA_TYPE_PROCESSING to DATA_TYPE_PROCESSING_HALF
//Optimized indexing can be turned off by changing the OPTIMIZED_INDEXING_SETTING value to 0 (not recommended; this slows down processing)
#define CURRENT_DATA_TYPE_PROCESSING DATA_TYPE_PROCESSING_FLOAT
#define OPTIMIZED_INDEXING_SETTING 1
#define USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT
#define CPU_OPTIMIZATION_SETTING USE_DEFAULT
#endif //(IMAGE_SET_TO_PROCESS == IMAGE_SET_PARAMETERS_FROM_PYTHON)

//remove (or don't use) capability for half precision if using GPU with compute capability under 5.3
#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE
typedef double beliefPropProcessingDataType;
#define BELIEF_PROP_PROCESSING_DATA_TYPE_STRING "DOUBLE"
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
typedef float beliefPropProcessingDataType;
#define BELIEF_PROP_PROCESSING_DATA_TYPE_STRING "FLOAT"
#endif

//number of belief propagation stereo runs of same image set
#define NUM_BP_STEREO_RUNS 15

#define INF_BP 65504.0f     // large cost (used for "infinity"), value set to support half type
#define SMALL_VAL_BP .01f

//define the default message value...
#define DEFAULT_INITIAL_MESSAGE_VAL 0.0f

//used to define the two checkerboard "parts" that the image is divided into
#define CHECKERBOARD_PART_1 1
#define CHECKERBOARD_PART_2 2

#define NO_EXPECTED_STEREO_BP -999.0f


//structure to store the settings for the number of levels and iterations
typedef struct
{
	int numLevels;
	int numIterations;

	int widthImages;
	int heightImages;

	float smoothingSigma;
	float lambda_bp;
	float data_k_bp;
	float disc_k_bp;
}BPsettings;

#endif /* BPSTEREOPARAMETERS_H_ */
