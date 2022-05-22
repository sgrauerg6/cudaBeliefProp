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
#include <numeric>
#include <algorithm>

//uncomment to only processes smaller stereo sets
//#define SMALLER_SETS_ONLY

const std::string BP_RUN_OUTPUT_FILE{"output.txt"};
const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{"outputResults.csv"};

//option to optimized thread block dimensions at each level by running BP w/ each thread block dimension in THREAD_DIMS_OPTIONS,
//finding the block dimensions with the lowest runtime at each level, and then setting each level to the optimized thread block
//dimensions in the final run
constexpr bool OPTIMIZE_THREAD_BLOCK_DIMS{true};
const std::vector<std::array<unsigned int, 2>> THREAD_DIMS_OPTIONS{
	{16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5}, {32, 6}, {32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4}, {128, 1}, {128, 2}, {256, 1}};

//get current CUDA properties and write them to output stream
void retrieveDeviceProperties(const int numDevice, std::ostream& resultsStream)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, numDevice);
	int cudaDriverVersion;
	cudaDriverGetVersion(&cudaDriverVersion);
	int cudaRuntimeVersion;
	cudaRuntimeGetVersion(&cudaRuntimeVersion);

	resultsStream << "Device " << numDevice << ": " << prop.name << " with " << prop.multiProcessorCount << " multiprocessors\n";
	resultsStream << "Cuda version: " << cudaDriverVersion << "\n";
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";
}

template<typename T, unsigned int NUM_SET>
void runBpOnSetAndUpdateResults(std::array<std::map<std::string, std::vector<std::string>>, 2>& resultsDefaultTBFinal,
								const bool isTemplatedDispVals)
{
	//load all the BP default settings as set in bpStereoCudaParameters.cuh
	BPsettings algSettings;
	algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

	//if optimizing thread block dimensions, threadDimsVect contains thread block dimension options (and is empty if not)
	const std::vector<std::array<unsigned int, 2>> threadDimsVect{OPTIMIZE_THREAD_BLOCK_DIMS ? THREAD_DIMS_OPTIONS : std::vector<std::array<unsigned int, 2>>()};
    std::vector<std::map<std::array<unsigned int, 2>, double>> tDimsToRuntimeEachLevel(algSettings.numLevels_);

	//currCudaParams initialized with default thread block dimensions at every level
	bp_cuda_params::CudaParameters currCudaParams(algSettings.numLevels_);
	
	//if optimizing thread block dimensions, run BP for each thread block option, retrieve best thread block dimensions at each level,
	//and then run BP with best found thread block dimensions at each level
	//if not optimizing thread block dimension, run BP once using default thread block dimensions
	for (unsigned int runNum=0; runNum <= threadDimsVect.size(); runNum++) {
		const std::array<unsigned int, 2>* tBlockDims = (runNum < threadDimsVect.size()) ? (&(threadDimsVect[runNum])) : nullptr;
		std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);
		retrieveDeviceProperties(0, resultsStream);
		if (runNum < threadDimsVect.size()) {
  		  //set thread block dimensions to current tBlockDims for each BP processing level
		  currCudaParams.blockDimsXY_ = std::vector<std::array<unsigned int, 2>>(algSettings.numLevels_, *tBlockDims);
		}

		resultsStream << "DataType:" << DATA_SIZE_TO_NAME_MAP.at(sizeof(T)) << std::endl;
		for (unsigned int level=0; level < algSettings.numLevels_; level++) {
		  resultsStream << "Level " << std::to_string(level) << " Thread Block Width:" << currCudaParams.blockDimsXY_[level][0] << std::endl;
		  resultsStream << "Level " << std::to_string(level) << " Thread Block Height:" << currCudaParams.blockDimsXY_[level][1] << std::endl;
		}
		std::array<std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>, 2> runBpStereo = {
				std::make_unique<RunBpStereoSetOnGPUWithCUDA<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>(currCudaParams),
				std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>()
		};

        //run optimized implementation only if not final run or run using default thread block size (those are the only runs that are output in final results)
		const bool runOptImpOnly{!(((runNum == threadDimsVect.size()) || ((*tBlockDims) ==
		                            std::array<unsigned int, 2>{bp_cuda_params::DEFAULT_BLOCK_SIZE_WIDTH_BP, bp_cuda_params::DEFAULT_BLOCK_SIZE_HEIGHT_BP})))};
		if (isTemplatedDispVals) {
			RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
					resultsStream, runBpStereo, NUM_SET, algSettings, runOptImpOnly);
		}
		else {
			std::unique_ptr<RunBpStereoSet<T, 0>> optCUDADispValsNoTemplate = std::make_unique<RunBpStereoSetOnGPUWithCUDA<T, 0>>(currCudaParams);
			RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>(
					resultsStream, optCUDADispValsNoTemplate, runBpStereo[1], NUM_SET, algSettings, runOptImpOnly);
		}
		resultsStream.close();

        //get resulting including runtimes for each BP level for current run
		const auto resultsCurrentRun = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).first;
		if (runNum < threadDimsVect.size()) {
			for (unsigned int level=0; level < algSettings.numLevels_; level++) {
				tDimsToRuntimeEachLevel[level][*tBlockDims] = std::stod(resultsCurrentRun.at("Level " + std::to_string(level) + " Runtime (" + 
				                                                       std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
			}
			if (runNum == (threadDimsVect.size() - 1)) {
				//retrieve and set optimized thread block dimensions at each level for final run
				//std::min_element used to retrieve thread block dimensions corresponding to lowest runtime from previous runs
				std::transform(tDimsToRuntimeEachLevel.begin(), tDimsToRuntimeEachLevel.end(), currCudaParams.blockDimsXY_.begin(),
				               [](const auto& tDimsToRunTimeCurrLevel) -> std::array<unsigned int, 2> { 
								   return (std::min_element(tDimsToRunTimeCurrLevel.begin(), tDimsToRunTimeCurrLevel.end(),
					                       [](const auto& a, const auto& b) { return a.second < b.second; }))->first; });
		    }
		}
		if ((runNum == threadDimsVect.size()) || 
		    ((*tBlockDims) == std::array<unsigned int, 2>{bp_cuda_params::DEFAULT_BLOCK_SIZE_WIDTH_BP, bp_cuda_params::DEFAULT_BLOCK_SIZE_HEIGHT_BP}))
	    {
			//only show output for run using default thread block dimensions and final run (which is the same run if not optimizing thread block size)
			auto& resultUpdate = (runNum == threadDimsVect.size()) ? resultsDefaultTBFinal[1] : resultsDefaultTBFinal[0];
			for (const auto& currRunResult : resultsCurrentRun) {
				if (resultUpdate.count(currRunResult.first)) {
					resultUpdate[currRunResult.first].push_back(currRunResult.second);
				}
				else {
					resultUpdate[currRunResult.first] = std::vector{currRunResult.second};
				}
			}
		}
	}
}

void getAverageMedianSpeedup(const std::array<std::map<std::string, std::vector<std::string>>, 2>& resultsDefaultTBFinal) {
	const std::string RUNTIME_HEADER{"Median CUDA runtime (including transfer time)"};
	unsigned int numRuns = resultsDefaultTBFinal[0].at(RUNTIME_HEADER).size();
	std::vector<double> speedupsVect;
	for (unsigned int i=0; i < numRuns; i++) {
		const double rTimeDefaultTB = std::stod(resultsDefaultTBFinal[0].at(RUNTIME_HEADER).at(i));
		const double rTimeOptTB = std::stod(resultsDefaultTBFinal[1].at(RUNTIME_HEADER).at(i));
		const double speedup = rTimeDefaultTB / rTimeOptTB;
		speedupsVect.push_back(speedup);
	}
	std::sort(speedupsVect.begin(), speedupsVect.end());
	std::cout << "Average speedup: " << (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)numRuns) << std::endl;
	const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
	    (speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
		speedupsVect[(speedupsVect.size() / 2)];
	std::cout << "Median speedup: " << medianSpeedup << std::endl;
}

int main(int argc, char** argv)
{
	//size_t heapSize;
	//cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize*50);
	std::array<std::map<std::string, std::vector<std::string>>, 2> resultsDefaultTBFinal;
	runBpOnSetAndUpdateResults<float, 0>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 0>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<float, 1>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 1>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<float, 2>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 2>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<float, 3>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 3>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<float, 4>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 4>(resultsDefaultTBFinal, false);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<float, 5>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 5>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<float, 6>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<float, 6>(resultsDefaultTBFinal, false);
#endif //SMALLER_SETS_ONLY
#ifdef DOUBLE_PRECISION_SUPPORTED
	runBpOnSetAndUpdateResults<double, 0>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 0>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<double, 1>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 1>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<double, 2>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 2>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<double, 3>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 3>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<double, 4>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 4>(resultsDefaultTBFinal, false);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<double, 5>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 5>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<double, 6>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<double, 6>(resultsDefaultTBFinal, false);
#endif //SMALLER_SETS_ONLY
#endif //DOUBLE_PRECISION_SUPPORTED
#ifdef CUDA_HALF_SUPPORT
	runBpOnSetAndUpdateResults<short, 0>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 0>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<short, 1>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 1>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<short, 2>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 2>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<short, 3>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 3>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<short, 4>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 4>(resultsDefaultTBFinal, false);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<short, 5>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 5>(resultsDefaultTBFinal, false);
	runBpOnSetAndUpdateResults<short, 6>(resultsDefaultTBFinal, true);
	runBpOnSetAndUpdateResults<short, 6>(resultsDefaultTBFinal, false);
#endif //SMALLER_SETS_ONLY
#endif //CUDA_HALF_SUPPORT
	const auto headersInOrder = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).second;
	if constexpr (OPTIMIZE_THREAD_BLOCK_DIMS) {
		getAverageMedianSpeedup(resultsDefaultTBFinal);
	}

	std::ofstream resultsStream(BP_ALL_RUNS_OUTPUT_CSV_FILE);
	for (const auto& currHeader : headersInOrder) {
		resultsStream << currHeader << ",";
	}
	resultsStream << std::endl;

    for (const auto& resultsAcrossRunsSet : resultsDefaultTBFinal) {
		for (unsigned int i=0; i < resultsAcrossRunsSet.begin()->second.size(); i++) {
			for (auto& currHeader : headersInOrder) {
				resultsStream << resultsAcrossRunsSet.at(currHeader).at(i) << ",";
			}
			resultsStream << std::endl;
		}
	}
	resultsStream.close();

	std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run in "
			  << BP_ALL_RUNS_OUTPUT_CSV_FILE << std::endl;
}
