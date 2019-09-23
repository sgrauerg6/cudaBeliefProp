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
#include "ParameterFiles/bpStereoParameters.h"
#include "ParameterFiles/bpStructsAndEnums.h"
#include "SingleThreadCPU/stereo.h"

//needed to run the optimized implementation a stereo set using CPU
#include "OptimizeCPU/RunBpStereoOptimizedCPU.h"

//needed to run the implementation a stereo set using CUDA
#include "OutputEvaluation/DisparityMap.h"
#include "OutputEvaluation/OutputEvaluationParameters.h"
#include "OutputEvaluation/OutputEvaluationResults.h"
#include "../FileProcessing/BpFileHandling.h"
#include <filesystem>

//compare resulting disparity map with a ground truth (or some other disparity map...)
//this function takes as input the file names of a two disparity maps and the factor
//that each disparity was scaled by in the generation of the disparity map image
void compareDispMaps(const DisparityMap<float>& outputDisparityMap, const DisparityMap<float>& groundTruthDisparityMap, std::ostream& resultsStream)
{
	const OutputEvaluationResults<float> outputEvalResults = outputDisparityMap.getOuputComparison(groundTruthDisparityMap, OutputEvaluationParameters<float>());
	resultsStream << outputEvalResults;
}

//run the CUDA stereo implementation on the default reference and test images with the result saved to the default
//saved disparity map file as defined in bpStereoCudaParameters.cuh
void runStereoOnDefaultImagesUsingDefaultSettings(std::ostream& resultsStream)
{
	BpFileHandling bpFileSettings(bp_params::STEREO_SET);
	std::filesystem::path refImagePath = bpFileSettings.getRefImagePath();
	std::filesystem::path testImagePath = bpFileSettings.getTestImagePath();
	std::filesystem::path outputDisp1 = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
	std::filesystem::path outputDisp2 = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
	std::filesystem::path groundTruthDisp = bpFileSettings.getGroundTruthDisparityFilePath();

	//load all the BP default settings as set in bpStereoCudaParameters.cuh
	BPsettings algSettings;

	std::cout << "Running belief propagation on reference image " << refImagePath << " and test image " << testImagePath << " on GPU and CPU\n";
	std::cout << "Output disparity map from final GPU run at " << outputDisp1 << "\n";
	std::cout << "Output disparity map from CPU run at " << outputDisp2 << "\n";	RunBpStereoOptimizedCPU<beliefPropProcessingDataType> runBpStereoOptCPU;
	RunBpStereoCPUSingleThread<beliefPropProcessingDataType> runBpStereoSetCPU;
	ProcessStereoSetOutput optCPURunTimeAndDispImage = runBpStereoOptCPU(refImagePath,
			testImagePath, algSettings, resultsStream);
	optCPURunTimeAndDispImage.outDisparityMap.saveDisparityMap(outputDisp1, bp_params::SCALE_BP);
	ProcessStereoSetOutput singleThreadCpuRunTimeAndDispImage = runBpStereoSetCPU(refImagePath,
			testImagePath, algSettings, resultsStream);
	singleThreadCpuRunTimeAndDispImage.outDisparityMap.saveDisparityMap(outputDisp2, bp_params::SCALE_BP);

	DisparityMap<float> groundTruthDisparityMap(groundTruthDisp, bp_params::SCALE_BP);

	resultsStream << "Median opt CPU runtime (including transfer time): " << optCPURunTimeAndDispImage.runTime << "\n";
	resultsStream << "Single Thread CPU runtime: " << singleThreadCpuRunTimeAndDispImage.runTime << "\n";


	resultsStream << "\nCPU output vs. Ground Truth result:\n";
	compareDispMaps(singleThreadCpuRunTimeAndDispImage.outDisparityMap, groundTruthDisparityMap, resultsStream);
	resultsStream << "\nOpt CPU output vs. Ground Truth result:\n";
	compareDispMaps(optCPURunTimeAndDispImage.outDisparityMap, groundTruthDisparityMap, resultsStream);
	resultsStream << "\nOpt CPU output vs. CPU output:\n";
	compareDispMaps(singleThreadCpuRunTimeAndDispImage.outDisparityMap, optCPURunTimeAndDispImage.outDisparityMap, resultsStream);

	std::cout << "More info including input parameters, detailed timings, and output disparity maps comparison to ground truth in output.txt.\n";
}


int main(int argc, char** argv)
{
	//std::ofstream resultsStream("output.txt", std::ofstream::out);
	std::ostream resultsStream(std::cout.rdbuf());
	int optLevel = USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT_FROM_PYTHON;
	resultsStream << "Ref Image: " << bp_params::DEFAULT_REF_IMAGE_PATH << "\n";
	resultsStream << "Test Image: " << bp_params::DEFAULT_TEST_IMAGE_PATH
			<< "\n";
	resultsStream << "Memory Optimization Level: " << optLevel << "\n";
	resultsStream << "Indexing Optimization Level: "
			<< OPTIMIZED_INDEXING_SETTING << "\n";
	resultsStream << "BP Processing Data Type: "
			<< BELIEF_PROP_PROCESSING_DATA_TYPE_STRING << "\n";
	resultsStream << "Num Possible Disparity Values: "
			<< NUM_POSSIBLE_DISPARITY_VALUES << "\n";
	resultsStream << "Num BP Levels: " << bp_params::LEVELS_BP << "\n";
	resultsStream << "Num BP Iterations: " << bp_params::ITER_BP << "\n";
	resultsStream << "DISC_K_BP: " << bp_params::DISC_K_BP << "\n";
	resultsStream << "DATA_K_BP: " << bp_params::DATA_K_BP << "\n";
	resultsStream << "LAMBDA_BP: " << bp_params::LAMBDA_BP << "\n";
	resultsStream << "SIGMA_BP: " << bp_params::SIGMA_BP << "\n";
	resultsStream << "CPU_OPTIMIZATION_LEVEL: " << CPU_OPTIMIZATION_SETTING
			<< "\n";
	resultsStream << "BYTES_ALIGN_MEMORY: " << BYTES_ALIGN_MEMORY << "\n";
	resultsStream << "NUM_DATA_ALIGN_WIDTH: " << NUM_DATA_ALIGN_WIDTH << "\n";
	runStereoOnDefaultImagesUsingDefaultSettings(resultsStream);
}
