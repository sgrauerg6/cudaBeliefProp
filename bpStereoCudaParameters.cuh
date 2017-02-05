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

#ifndef BP_STEREO_CUDA_PARAMETERS_CUH
#define BP_STEREO_CUDA_PARAMETERS_CUH

#include <stdio.h>

#define TSUKUBA_IMAGES 1
#define CONES_IMAGES_QUARTER_SIZE 2
#define CONES_IMAGES_HALF_SIZE 3
#define CONES_IMAGES_FULL_SIZE 4
#define IMAGE_SET_TO_PROCESS TSUKUBA_IMAGES

#if (IMAGE_SET_TO_PROCESS == TSUKUBA_IMAGES)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "refImageTsukuba.pgm"
#define DEFAULT_TEST_IMAGE_PATH "testImageTsukuba.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_GPU "computedDisparityMapTsukubaGPU.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_CPU "computedDisparityMapTsukubaCPU.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 15

#define SCALE_BP 16.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "groundTruthDispTsukuba.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 16.0f //scaling from ground truth disparity to ground truth disparity map image

#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_QUARTER_SIZE)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "conesQuarter2.pgm"
#define DEFAULT_TEST_IMAGE_PATH "conesQuarter6.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_GPU "computedDisparityConesQuarterGPU.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_CPU "computedDisparityConesQuarterCPU.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 63

#define SCALE_BP 4.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "conesQuarterGroundTruth.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 4.0f //scaling from ground truth disparity to ground truth disparity map image

#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_HALF_SIZE)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "conesHalf2.pgm"
#define DEFAULT_TEST_IMAGE_PATH "conesHalf6.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_GPU "computedDisparityConesHalfGPU.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_CPU "computedDisparityConesHalfCPU.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 110

#define SCALE_BP 2.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "conesHalfGroundTruth.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 2.0f //scaling from ground truth disparity to ground truth disparity map image

#elif (IMAGE_SET_TO_PROCESS == CONES_IMAGES_FULL_SIZE)

//define the path for the 'default' reference and test images and the output "movement" images (can easily run
//on other images using runBpStereoImageSeries on any number of images)
#define DEFAULT_REF_IMAGE_PATH "conesFull2.pgm"
#define DEFAULT_TEST_IMAGE_PATH "conesFull6.pgm"

#define SAVE_DISPARITY_IMAGE_PATH_GPU "computedDisparityConesFullGPU.pgm"
#define SAVE_DISPARITY_IMAGE_PATH_CPU "computedDisparityConesFullCPU.pgm"

//defines the possible number of disparity values (range is from 0 to (NUM_POSSIBLE_DISPARITY_VALUES - 1) in increments of 1)
#define NUM_POSSIBLE_DISPARITY_VALUES 255

#define SCALE_BP 1.0f     // scaling from computed disparity to graylevel in output

//info about a default ground truth
#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE "conesFullGroundTruth.pgm"
#define DEFAULT_SCALE_GROUND_TRUTH_DISPARITY 1.0f //scaling from ground truth disparity to ground truth disparity map image

#endif //IMAGE_SET_TO_PROCESS

//number of belief propagation stereo runs of same image set
#define NUM_BP_STEREO_RUNS 5

#define INF_BP 100000000.0f     // large cost (used for "infinity")
#define SMALL_VAL_BP .01f

//define the default message value...
#define DEFAULT_INITIAL_MESSAGE_VAL 0.0f

// number of BP iterations at each scale/level
#define ITER_BP 100

// number of scales/levels in the pyramid to run BP
#define LEVELS_BP 1

//truncation of discontinuity cost
#define DISC_K_BP 1.7f

// truncation of data cost
#define DATA_K_BP 15.0f  

// weighing of data cost
#define LAMBDA_BP 0.07f

#define MIN_SIGMA_VAL_SMOOTH 0.1f //don't smooth input images if SIGMA_BP below this
#define SIGMA_BP 1.0f    // amount to smooth the input images

#define DEFAULT_X_BORDER_GROUND_TRUTH_DISPARITY 18
#define DEFAULT_Y_BORDER_GROUND_TRUTH_DISPARITY 18

//more parameters for smoothing
#define WIDTH_SIGMA_1 4.0
#define MAX_SIZE_FILTER 25

//defines the width and height of the thread block used for 
//image filtering (applying the Guassian filter in smoothImageHost)
#define BLOCK_SIZE_WIDTH_FILTER_IMAGES 16
#define BLOCK_SIZE_HEIGHT_FILTER_IMAGES 16

//defines the width and height of the thread block used for 
//each kernal function when running BP (right now same thread 
//block dimensions are used for each kernal function when running
//kernal function in runBpStereoHost.cu, though this could be
//changed)
#define BLOCK_SIZE_WIDTH_BP 32
#define BLOCK_SIZE_HEIGHT_BP 4

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
	
	float dataWeight;
	float dataCostCap;
	
	float discCostCap;
}BPsettings;




#endif // BP_STEREO_CUDA_PARAMETERS_CUH
