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
#include "bpStereoCudaParameters.h"
#include "SingleThreadCPU/stereo.h"

//needed for functions to load input images/store resulting disp/movement image
#include "imageHelpers.h"

//needed to evaluate the disparity/Stereo found
#include "stereoResultsEval.h"

//needed to run the implementation a stereo set using CUDA
#include "OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
//#include "OptimizeCPU/RunBpStereoOptimizedCPU.h"

//compare resulting disparity map with a ground truth (or some other disparity map...)
//this function takes as input the file names of a two disparity maps and the factor
//that each disparity was scaled by in the generation of the disparity map image
void compareDispMaps(const char* dispMap1File, float scaleComputedDispMap, const char* dispMap2File, float scaleGroundTruthDispMap, FILE* resultsFile)
{
	unsigned int widthDispMap = 0;
	unsigned int heightDispMap = 0;

	//first retrieve the unsigned int arrays from the computed disparity map and ground truth disparity map images
	unsigned int* compDispMapUnsignedInts = ImageHelperFunctions::loadImageFromPGM(dispMap1File, widthDispMap, heightDispMap);
	unsigned int* groundTruthDispMapUnsignedInts = ImageHelperFunctions::loadImageFromPGM(dispMap2File, widthDispMap, heightDispMap);

	//retrieve the evaluation between the two disparity maps according to the parameters in stereoResultsEvalParameters.cuh
	StereoResultsEvaluation runStereoEvaluation;
	stereoEvaluationResults* stereoEvaluation =
			runStereoEvaluation.runStereoResultsEvaluationUsedUnsignedIntScaledDispMap(compDispMapUnsignedInts, groundTruthDispMapUnsignedInts, scaleComputedDispMap, scaleGroundTruthDispMap, widthDispMap, heightDispMap);

	runStereoEvaluation.writeStereoEvaluationResultsToFile(stereoEvaluation, resultsFile);
}


BPsettings initializeAndReturnBPSettings()
{
	BPsettings startBPSettings;

	startBPSettings.smoothingSigma = SIGMA_BP;
	startBPSettings.numLevels = LEVELS_BP;
	startBPSettings.numIterations = ITER_BP;
	startBPSettings.lambda_bp = LAMBDA_BP;
	startBPSettings.data_k_bp = DATA_K_BP;
	startBPSettings.disc_k_bp = DISC_K_BP;

	//height/width determined when image read from file
	startBPSettings.widthImages = 0;
	startBPSettings.heightImages = 0;

	return startBPSettings;
}


//run the CUDA stereo implementation on the default reference and test images with the result saved to the default
//saved disparity map file as defined in bpStereoCudaParameters.cuh
void runStereoOnDefaultImagesUsingDefaultSettings(FILE* resultsFile)
{
	//load all the BP default settings as set in bpStereoCudaParameters.cuh
	BPsettings algSettings = initializeAndReturnBPSettings();

	printf(
			"Running belief propagation on reference image %s and test image %s on GPU and CPU\n",
			DEFAULT_REF_IMAGE_PATH, DEFAULT_TEST_IMAGE_PATH);
	RunBpStereoSetOnGPUWithCUDA<beliefPropProcessingDataType> runBpStereoSetCUDA;
	//RunBpStereoOptimizedCPU<beliefPropProcessingDataType> runBpStereoOptCPU;
	RunBpStereoCPUSingleThread<beliefPropProcessingDataType> runBpStereoSetCPU;
	float cudaRunTime = runBpStereoSetCUDA(DEFAULT_REF_IMAGE_PATH,
			DEFAULT_TEST_IMAGE_PATH, algSettings, SAVE_DISPARITY_IMAGE_PATH_1,
			resultsFile);
	float singleThreadCpuRunTime = runBpStereoSetCPU(DEFAULT_REF_IMAGE_PATH,
			DEFAULT_TEST_IMAGE_PATH, algSettings, SAVE_DISPARITY_IMAGE_PATH_2,
			resultsFile);
	printf("Single Thread CPU runtime: %f\n", singleThreadCpuRunTime);
	/*float cpuRunTime = runBpStereoOptCPU(DEFAULT_REF_IMAGE_PATH,
			DEFAULT_TEST_IMAGE_PATH, algSettings, SAVE_DISPARITY_IMAGE_PATH_2,
			resultsFile);*/

	printf("Median CUDA runtime (including transfer time): %f\n", cudaRunTime);
	//printf("Optimized CPU runtime: %f\n", cpuRunTime);
	printf("Single Thread CPU runtime: %f\n", singleThreadCpuRunTime);
	printf("Output disparity map from final GPU run at %s\n",
			SAVE_DISPARITY_IMAGE_PATH_1);
	printf("Output disparity map from CPU run at %s\n",
			SAVE_DISPARITY_IMAGE_PATH_2);

	fprintf(resultsFile, "\nCPU output vs. Ground Truth result:\n");
	compareDispMaps(SAVE_DISPARITY_IMAGE_PATH_2, SCALE_BP,
			DEFAULT_GROUND_TRUTH_DISPARITY_FILE,
			DEFAULT_SCALE_GROUND_TRUTH_DISPARITY, resultsFile);
	fprintf(resultsFile, "\nGPU output vs. Ground Truth result:\n");
	compareDispMaps(SAVE_DISPARITY_IMAGE_PATH_1, SCALE_BP,
			DEFAULT_GROUND_TRUTH_DISPARITY_FILE,
			DEFAULT_SCALE_GROUND_TRUTH_DISPARITY, resultsFile);
	fprintf(resultsFile, "\nGPU output vs. CPU output:\n");
	compareDispMaps(SAVE_DISPARITY_IMAGE_PATH_2, SCALE_BP,
			SAVE_DISPARITY_IMAGE_PATH_1, DEFAULT_SCALE_GROUND_TRUTH_DISPARITY,
			resultsFile);
	printf(
			"More info including input parameters, detailed timings, and output disparity maps comparison to ground truth are in output.txt file.\n");
}


void retrieveDeviceProperties(int numDevice, FILE* resultsFile)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, numDevice);
	int cudaDriverVersion;
	cudaDriverGetVersion(&cudaDriverVersion);

	fprintf(resultsFile, "Device %d: %s with %d multiprocessors\n", numDevice, prop.name, prop.multiProcessorCount);
	fprintf(resultsFile, "Cuda version: %d\n", cudaDriverVersion);
}


int main(int argc, char** argv)
{
	//FILE* resultsFile = stdout;
	FILE* resultsFile = fopen("output.txt", "w");
	int optLevel = USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT_FROM_PYTHON;
	fprintf(resultsFile, "Ref Image: %s\n", DEFAULT_REF_IMAGE_PATH);
	fprintf(resultsFile, "Test Image: %s\n", DEFAULT_TEST_IMAGE_PATH);
	fprintf(resultsFile, "Memory Optimization Level: %d\n", optLevel);
	fprintf(resultsFile, "Indexing Optimization Level: %d\n", OPTIMIZED_INDEXING_SETTING);
	fprintf(resultsFile, "BP Processing Data Type: %s\n", BELIEF_PROP_PROCESSING_DATA_TYPE_STRING);
	fprintf(resultsFile, "Num Possible Disparity Values: %d\n", NUM_POSSIBLE_DISPARITY_VALUES);
	fprintf(resultsFile, "Num BP Levels: %d\n", LEVELS_BP);
	fprintf(resultsFile, "Num BP Iterations: %d\n", ITER_BP);
	fprintf(resultsFile, "DISC_K_BP: %f\n", DISC_K_BP);
	fprintf(resultsFile, "DATA_K_BP: %f\n", DATA_K_BP);
	fprintf(resultsFile, "LAMBDA_BP: %f\n", LAMBDA_BP);
	fprintf(resultsFile, "SIGMA_BP: %f\n", SIGMA_BP);
	fprintf(resultsFile, "CPU_OPTIMIZATION_LEVEL: %d\n", CPU_OPTIMIZATION_SETTING);
	fprintf(resultsFile, "BYTES_ALIGN_MEMORY: %d\n", BYTES_ALIGN_MEMORY);
	fprintf(resultsFile, "NUM_DATA_ALIGN_WIDTH: %d\n", NUM_DATA_ALIGN_WIDTH);
	retrieveDeviceProperties(0, resultsFile);
	runStereoOnDefaultImagesUsingDefaultSettings(resultsFile);
	int cudaRuntimeVersion;
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	fprintf(resultsFile, "Cuda Runtime Version: %d\n", cudaRuntimeVersion);
}
