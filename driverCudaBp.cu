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

//needed to run the implementation a stereo set using CUDA
#include "OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
#include "OutputEvaluation/DisparityMap.h"
#include "OutputEvaluation/OutputEvaluationParameters.h"
#include "OutputEvaluation/OutputEvaluationResults.h"

#include <fstream>

//compare resulting disparity map with a ground truth (or some other disparity map...)
//this function takes as input the file names of a two disparity maps and the factor
//that each disparity was scaled by in the generation of the disparity map image
void compareDispMaps(const DisparityMap<float>& outputDisparityMap, const DisparityMap<float>& groundTruthDisparityMap, std::ostream& resultsStream)
{
	const OutputEvaluationResults<float> outputEvalResults = outputDisparityMap.getOuputComparison(groundTruthDisparityMap, OutputEvaluationParameters<float>());
	resultsStream << outputEvalResults;
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

	return startBPSettings;
}

//run the CUDA stereo implementation on the default reference and test images with the result saved to the default
//saved disparity map file as defined in bpStereoCudaParameters.cuh
void runStereoOnDefaultImagesUsingDefaultSettings(std::ostream& outStream)
{
	//load all the BP default settings as set in bpStereoCudaParameters.cuh
	BPsettings algSettings = initializeAndReturnBPSettings();

	std::cout << "Running belief propagation on reference image " << DEFAULT_REF_IMAGE_PATH << " and test image " << DEFAULT_TEST_IMAGE_PATH << " on GPU and CPU\n";
	RunBpStereoSetOnGPUWithCUDA<beliefPropProcessingDataType> runBpStereoSetCUDA;
	RunBpStereoCPUSingleThread<beliefPropProcessingDataType> runBpStereoSetCPU;
	ProcessStereoSetOutput cudaRunTimeAndDispImage = runBpStereoSetCUDA(DEFAULT_REF_IMAGE_PATH,
			DEFAULT_TEST_IMAGE_PATH, algSettings, outStream);
	cudaRunTimeAndDispImage.outDisparityMap.saveDisparityMap(SAVE_DISPARITY_IMAGE_PATH_1, SCALE_BP);
	ProcessStereoSetOutput singleThreadCpuRunTimeAndDispImage = runBpStereoSetCPU(DEFAULT_REF_IMAGE_PATH,
			DEFAULT_TEST_IMAGE_PATH, algSettings, outStream);
	singleThreadCpuRunTimeAndDispImage.outDisparityMap.saveDisparityMap(SAVE_DISPARITY_IMAGE_PATH_2, SCALE_BP);

	DisparityMap<float> groundTruthDisparityMap(DEFAULT_GROUND_TRUTH_DISPARITY_FILE, (unsigned int)SCALE_BP);

	outStream << "Median CUDA runtime (including transfer time): " << cudaRunTimeAndDispImage.runTime << std::endl;
	outStream << "Single Thread CPU runtime: " << singleThreadCpuRunTimeAndDispImage.runTime << std::endl;
	std::cout << "Output disparity map from final GPU run at " << SAVE_DISPARITY_IMAGE_PATH_1 << std::endl;
	std::cout << "Output disparity map from CPU run at " << SAVE_DISPARITY_IMAGE_PATH_2 << std::endl;

	outStream << "\nCPU output vs. Ground Truth result:\n";
	compareDispMaps(singleThreadCpuRunTimeAndDispImage.outDisparityMap, groundTruthDisparityMap, outStream);
	outStream << "\nGPU output vs. Ground Truth result:\n";
	compareDispMaps(cudaRunTimeAndDispImage.outDisparityMap, groundTruthDisparityMap, outStream);
	outStream << "\nGPU output vs. CPU output:\n";
	compareDispMaps(singleThreadCpuRunTimeAndDispImage.outDisparityMap, cudaRunTimeAndDispImage.outDisparityMap, outStream);

	std::cout << "More info including input parameters, detailed timings, and output disparity maps comparison to ground truth in output.txt.\n";
}


void retrieveDeviceProperties(int numDevice, std::ostream& resultsStream)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, numDevice);
	int cudaDriverVersion;
	cudaDriverGetVersion(&cudaDriverVersion);

	resultsStream << "Device " << numDevice << ": " << prop.name << " with " << prop.multiProcessorCount << " multiprocessors\n";
	resultsStream << "Cuda version: " << cudaDriverVersion << "\n";
}


int main(int argc, char** argv)
{
	std::ofstream resultsStream("output.txt", std::ofstream::out);
	//std::ostream resultsStream(std::cout.rdbuf());
	int optLevel = USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT_FROM_PYTHON;
	resultsStream << "Ref Image: " << DEFAULT_REF_IMAGE_PATH << "\n";
	resultsStream << "Test Image: " << DEFAULT_TEST_IMAGE_PATH << "\n";
	resultsStream << "Memory Optimization Level: " << optLevel << "\n";
	resultsStream << "Indexing Optimization Level: " << OPTIMIZED_INDEXING_SETTING << "\n";
	resultsStream << "BP Processing Data Type: " << BELIEF_PROP_PROCESSING_DATA_TYPE_STRING << "\n";
	resultsStream << "Num Possible Disparity Values: " << NUM_POSSIBLE_DISPARITY_VALUES << "\n";
	resultsStream << "Num BP Levels: " << LEVELS_BP << "\n";
	resultsStream << "Num BP Iterations: " << ITER_BP << "\n";
	resultsStream << "DISC_K_BP: " << DISC_K_BP << "\n";
	resultsStream << "DATA_K_BP: " << DATA_K_BP << "\n";
	resultsStream << "LAMBDA_BP: " << LAMBDA_BP << "\n";
	resultsStream << "SIGMA_BP: " << SIGMA_BP << "\n";
	resultsStream << "CPU_OPTIMIZATION_LEVEL: " << CPU_OPTIMIZATION_SETTING << "\n";
	resultsStream << "BYTES_ALIGN_MEMORY: " << BYTES_ALIGN_MEMORY << "\n";
	resultsStream << "NUM_DATA_ALIGN_WIDTH: " << NUM_DATA_ALIGN_WIDTH << "\n";
	resultsStream << "USE_SHARED_MEMORY: " << USE_SHARED_MEMORY << "\n";
	resultsStream << "DISP_INDEX_START_REG_LOCAL_MEM: " << DISP_INDEX_START_REG_LOCAL_MEM << "\n";
	retrieveDeviceProperties(0, resultsStream);
	runStereoOnDefaultImagesUsingDefaultSettings(resultsStream);
	int cudaRuntimeVersion;
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";
}
