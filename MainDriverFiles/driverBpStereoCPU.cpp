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

#include "../SingleThreadCPU/stereo.h"

//needed to run the optimized implementation a stereo set using CPU
#include "../OptimizeCPU/RunBpStereoOptimizedCPU.h"

#include "RunAndEvaluateBpResults.h"
#include <memory>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

const std::string BP_RUN_OUTPUT_FILE{"output.txt"};
const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{"outputResults.csv"};

template<typename T, unsigned int NUM_SET>
void runBpOnSetAndUpdateResults(const std::string& dataTypeName, std::map<std::string, std::vector<std::string>>& resultsAcrossRuns,
		const bool isTemplatedDispVals)
{
	std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);

	std::array<std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>, 2> runBpStereo = {
			std::make_unique<RunBpStereoOptimizedCPU<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>()
	};

	//load all the BP default settings as set in bpStereoCudaParameters.cuh
	BPsettings algSettings;
	algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

	resultsStream << "DataType:" << dataTypeName << std::endl;
	if (isTemplatedDispVals) {
		RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
				resultsStream, runBpStereo, NUM_SET, algSettings);
	}
	else {
		std::unique_ptr<RunBpStereoSet<T, 0>> optCpuDispValsNoTemplate = std::make_unique<RunBpStereoOptimizedCPU<T, 0>>();
		RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
				resultsStream, optCpuDispValsNoTemplate, runBpStereo[1], NUM_SET, algSettings);
	}
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
	runBpOnSetAndUpdateResults<float, 0>("FLOAT", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float, 0>("FLOAT", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float, 1>("FLOAT", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float, 1>("FLOAT", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float, 2>("FLOAT", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float, 2>("FLOAT", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float, 3>("FLOAT", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float, 3>("FLOAT", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float, 4>("FLOAT", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float, 4>("FLOAT", resultsAcrossRuns, false);
	/*runBpOnSetAndUpdateResults<double, 0>("DOUBLE", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<double, 0>("DOUBLE", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<double, 1>("DOUBLE", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<double, 1>("DOUBLE", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<double, 2>("DOUBLE", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<double, 2>("DOUBLE", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<double, 3>("DOUBLE", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<double, 3>("DOUBLE", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<double, 4>("DOUBLE", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<double, 4>("DOUBLE", resultsAcrossRuns, false);*/
	//runBpOnSetAndUpdateResults<float, 5>("FLOAT", resultsAcrossRuns, true);
	//runBpOnSetAndUpdateResults<float, 5>("FLOAT", resultsAcrossRuns, false);
#ifdef COMPILING_FOR_ARM
	runBpOnSetAndUpdateResults<float16_t, 0>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float16_t, 0>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float16_t, 1>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float16_t, 1>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float16_t, 2>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float16_t, 2>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float16_t, 3>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float16_t, 3>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<float16_t, 4>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<float16_t, 4>("HALF", resultsAcrossRuns, false);
	//runBpOnSetAndUpdateResults<float16_t, 5>("HALF", resultsAcrossRuns, true);
	//runBpOnSetAndUpdateResults<float16_t, 5>("HALF", resultsAcrossRuns, false);
#else
	runBpOnSetAndUpdateResults<short, 0>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<short, 0>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<short, 1>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<short, 1>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<short, 2>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<short, 2>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<short, 3>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<short, 3>("HALF", resultsAcrossRuns, false);
	runBpOnSetAndUpdateResults<short, 4>("HALF", resultsAcrossRuns, true);
	runBpOnSetAndUpdateResults<short, 4>("HALF", resultsAcrossRuns, false);
	//runBpOnSetAndUpdateResults<short, 5>("HALF", resultsAcrossRuns, true);
	//runBpOnSetAndUpdateResults<short, 5>("HALF", resultsAcrossRuns, false);
#endif

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

	std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run in "
			  << BP_ALL_RUNS_OUTPUT_CSV_FILE << std::endl;
}
