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
#include "../ParameterFiles/bpStereoCudaParameters.h"
#include "../SingleThreadCPU/stereo.h"

//needed to run the implementation a stereo set using CUDA
#include "../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
#include "RunAndEvaluateBpResults.h"
#include <memory>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <utility>

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

	//resultsStream << "USE_SHARED_MEMORY: " << USE_SHARED_MEMORY << "\n";
	//resultsStream << "DISP_INDEX_START_REG_LOCAL_MEM: " << DISP_INDEX_START_REG_LOCAL_MEM << "\n";
	int cudaRuntimeVersion;
	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>, 2> bpProcess_stereoSet0_float = {
			std::make_unique<RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>()
	};

	resultsStream << "DataType: FLOAT" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
			resultsStream, bpProcess_stereoSet0_float, 0);

	resultsStream.close();

	std::map<std::string, std::vector<std::string>> resultsAcrossRuns;
	auto resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile("output.txt").first;
	for (auto& currRunResult : resultsCurrentRun) {
		resultsAcrossRuns[currRunResult.first] = std::vector{currRunResult.second};
	}
	resultsStream.open("output.txt", std::ofstream::out);

	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>, 2> bpProcess_stereoSet1_float = {
			std::make_unique<RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>()
	};

	resultsStream << "DataType: FLOAT" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
			resultsStream, bpProcess_stereoSet1_float, 1);

	resultsStream.close();

	resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile("output.txt").first;
	for (auto& currRunResult : resultsCurrentRun) {
		resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
	}
	resultsStream.open("output.txt", std::ofstream::out);

	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>, 2> bpProcess_stereoSet2_float = {
			std::make_unique<RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>()
	};

	resultsStream << "DataType: FLOAT" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
			resultsStream, bpProcess_stereoSet2_float, 2);

	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	resultsStream.close();

#ifdef CUDA_HALF_SUPPORT

	resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile("output.txt").first;
	for (auto& currRunResult : resultsCurrentRun) {
		resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
	}

	resultsStream.open("output.txt", std::ofstream::out);

	std::array<std::unique_ptr<RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>, 2> bpProcess_stereoSet0_half = {
				std::make_unique<RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>(),
				std::make_unique<RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>()
	};

	resultsStream << "DataType: HALF" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(
			resultsStream, bpProcess_stereoSet0_half, 0);

	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	resultsStream.close();

	resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile("output.txt").first;
	for (auto& currRunResult : resultsCurrentRun) {
		resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
	}
	resultsStream.open("output.txt", std::ofstream::out);

	std::array<std::unique_ptr<RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>, 2> bpProcess_stereoSet1_half = {
			std::make_unique<RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>()
	};

	resultsStream << "DataType: HALF" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(
			resultsStream, bpProcess_stereoSet1_half, 1);

	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	resultsStream.close();

	resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile("output.txt").first;
	for (auto& currRunResult : resultsCurrentRun) {
		resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
	}
	resultsStream.open("output.txt", std::ofstream::out);

	std::array<std::unique_ptr<RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>, 2> bpProcess_stereoSet2_half = {
			std::make_unique<RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>()
	};

	resultsStream << "DataType: HALF" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(
			resultsStream, bpProcess_stereoSet2_half, 2);

	retrieveDeviceProperties(0, resultsStream);
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	resultsStream.close();

#endif //CUDA_HALF_SUPPORT

	std::vector<std::string> headersInOrder;
	std::tie(resultsCurrentRun, headersInOrder) = RunAndEvaluateBpResults::getResultsMappingFromFile("output.txt");
	for (auto& currRunResult : resultsCurrentRun) {
		resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
	}
	resultsStream.open("outputResults.csv", std::ofstream::out);

	for (auto& currHeader : headersInOrder) {
		resultsStream << currHeader << ",";
	}
	resultsStream << std::endl;

	for (unsigned int i=0; i < resultsAcrossRuns.begin()->second.size(); i++) {
		for (auto& currHeader : headersInOrder) {
			resultsStream << resultsAcrossRuns[currHeader][i] << ",";
		}
		resultsStream << std::endl;
	}
	resultsStream.close();
}
