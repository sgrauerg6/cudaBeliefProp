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

const std::string BP_RUN_OUTPUT_FILE{"output.txt"};
const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{"outputResults.csv"};

void retrieveDeviceProperties(int numDevice, std::ostream& resultsStream)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, numDevice);
	int cudaDriverVersion;
	cudaDriverGetVersion(&cudaDriverVersion);

	resultsStream << "Device " << numDevice << ": " << prop.name << " with " << prop.multiProcessorCount << " multiprocessors\n";
	resultsStream << "Cuda version: " << cudaDriverVersion << "\n";
}

template<typename T, unsigned int NUM_SET>
void runBpOnSetAndUpdateResults(const std::string& dataTypeName, std::map<std::string, std::vector<std::string>>& resultsAcrossRuns)
{
	std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);
	retrieveDeviceProperties(0, resultsStream);
	int cudaRuntimeVersion;
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	std::array<std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>, 2> runBpStereo = {
			std::make_unique<RunBpStereoSetOnGPUWithCUDA<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>()
	};

	resultsStream << "DataType:" << dataTypeName << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
			resultsStream, runBpStereo, NUM_SET);
	resultsStream.close();

	auto resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).first;
	for (auto& currRunResult : resultsCurrentRun) {
		if (resultsAcrossRuns.count(currRunResult.first)) {
			resultsAcrossRuns[currRunResult.first].push_back(currRunResult.second);
		}
		else {
			resultsAcrossRuns[currRunResult.first] = std::vector{currRunResult.second};
		}
	}
}

int main(int argc, char** argv)
{
	std::map<std::string, std::vector<std::string>> resultsAcrossRuns;
	runBpOnSetAndUpdateResults<float, 0>("FLOAT", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<float, 1>("FLOAT", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<float, 2>("FLOAT", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<float, 3>("FLOAT", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<float, 4>("FLOAT", resultsAcrossRuns);
#ifdef CUDA_HALF_SUPPORT
	runBpOnSetAndUpdateResults<short, 0>("HALF", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<short, 1>("HALF", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<short, 2>("HALF", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<short, 3>("HALF", resultsAcrossRuns);
	runBpOnSetAndUpdateResults<short, 4>("HALF", resultsAcrossRuns);
#endif //CUDA_HALF_SUPPORT
	const auto headersInOrder = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).second;

	std::ofstream resultsStream(BP_ALL_RUNS_OUTPUT_CSV_FILE);
	for (const auto& currHeader : headersInOrder) {
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
