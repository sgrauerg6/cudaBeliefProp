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

//This file contains the "main" function that drives the CUDA BP implementation


//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "bpStereoCudaParameters.cuh"

//needed for functions to load input images/store resulting disp/movement image
#include "imageHelpersHost.cu"

//needed to use the CUDA implementation of the Gaussian filter to smooth the images
#include "smoothImageHost.cu"

//need to run the CUDA BP Stereo estimation implementation on the smoothed input images
#include "runBpStereoHost.cu"

//needed to evaluate the disparity/Stereo found
#include "stereoResultsEvalHost.cu"

//needed to run the "chunk-based" Stereo estimation
#include "runBpStereoDivideImage.cu"

//needed to run the implementation on a series of images
#include "runBpStereoImageSeries.cu"

//needed to save the resulting Stereo...
#include "saveResultingDisparityMap.cu"

//needed for general utility functions to evaluate the results
#include "utilityFunctsForEval.cu"


//compare resulting disparity map with a ground truth (or some other disparity map...)
//this function takes as input the file names of a computed disparity map and also the
//ground truth disparity map and the factor that each disparity was scaled by in the generation
//of the disparity map image
void compareComputedDispMapWithGroundTruth(const char* computedDispMapFile, float scaleComputedDispMap, const char* groundTruthDispMapFile, float scaleGroundTruthDispMap, unsigned int widthDispMap, unsigned int heightDispMap)
{
	printf("DISP MAP DIMS: %d %d\n", widthDispMap, heightDispMap);
	//first retrieve the unsigned int arrays from the computed disparity map and ground truth disparity map images
	unsigned int* compDispMapUnsignedInts = loadImageFromPGM(computedDispMapFile, widthDispMap, heightDispMap);
	printf("DISP MAP DIMS: %d %d\n", widthDispMap, heightDispMap);
	unsigned int* groundTruthDispMapUnsignedInts = loadImageFromPGM(groundTruthDispMapFile, widthDispMap, heightDispMap);

	//retrieve the evaluation between the two disparity maps according to the parameters in stereoResultsEvalParameters.cuh
	stereoEvaluationResults* stereoEvaluation =
		runStereoResultsEvaluationUsedUnsignedIntScaledDispMap(compDispMapUnsignedInts, groundTruthDispMapUnsignedInts, scaleComputedDispMap, scaleGroundTruthDispMap, widthDispMap, heightDispMap);

	printStereoEvaluationResults(stereoEvaluation);
}

//run the CUDA stereo implementation on the default reference and test images with the result saved to the default
//saved disparity map file as defined in bpStereoCudaParameters.cuh
void runStereoOnDefaultImagesUsingDefaultSettings()
{
	//load all the BP default settings as set in bpStereoCudaParameters.cuh
	BPsettings algSettings = initializeAndReturnBPSettings();


	//default image sequence has two images...reference image followed by the test image
	int numImagesInDefaultSequence = 2;

	const char* imageFiles[] = {DEFAULT_REF_IMAGE_PATH, DEFAULT_TEST_IMAGE_PATH};

	//only one set of images to save disparity map for...
	const char* saveDisparityMapFilePaths[] = {SAVE_DISPARITY_IMAGE_PATH};

	//do save resulting disparity map...
	bool saveResultingDisparityMap = true;

	runStereoEstOnImageSeries(imageFiles, numImagesInDefaultSequence, DEFAULT_WIDTH_IMAGES, DEFAULT_HEIGHT_IMAGES, algSettings, saveResultingDisparityMap, saveDisparityMapFilePaths);

	compareComputedDispMapWithGroundTruth(SAVE_DISPARITY_IMAGE_PATH, SCALE_BP, DEFAULT_GROUND_TRUTH_DISPARITY_FILE, DEFAULT_SCALE_GROUND_TRUTH_DISPARITY, DEFAULT_WIDTH_IMAGES, DEFAULT_HEIGHT_IMAGES);
}

void retrieveDeviceProperties(int numDevice)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, numDevice);

	printf("Device %d: %s with %d multiprocessors\n", numDevice, prop.name, prop.multiProcessorCount);
}


int main(int argc, char** argv)
{
	runStereoOnDefaultImagesUsingDefaultSettings();
}
