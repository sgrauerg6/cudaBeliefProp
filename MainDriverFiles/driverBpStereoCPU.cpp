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

//This file contains the "main" function that drives the optimized CPU BP implementation

#include "../SingleThreadCPU/stereo.h"

//needed to run the optimized implementation a stereo set using CPU
#include "../OptimizeCPU/RunBpStereoOptimizedCPU.h"

#include <memory>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <numeric>
#include <algorithm>

#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//uncomment to only process smaller stereo sets
//#define SMALLER_SETS_ONLY

const std::string BP_RUN_OUTPUT_FILE{"output.txt"};
const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{"outputResults.csv"};
const std::string BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE{"outputResultsDefaultParallelParams.csv"};

//option to optimize parallel parameters by running BP w/ multiple parallel parameters options by
//finding the parallel parameters with the lowest runtime, and then setting the parallel parameters
//to the best found parallel parameters in the final run
constexpr bool OPTIMIZE_PARALLEL_PARAMS{true};

//default setting is to use the same parallel parameters for all kernels in run
//testing on i7-11800H has found that using different parallel parameters (corresponding to OpenMP thread counts)
//in different kernels in the optimized CPU implementation can increase runtime (may want to test on additional processors)
constexpr beliefprop::OptParallelParamsSetting optParallelParamsSetting{beliefprop::OptParallelParamsSetting::SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN};

const unsigned int NUM_THREADS_CPU{std::thread::hardware_concurrency()};
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{
	{ NUM_THREADS_CPU, 1}, { (3 * NUM_THREADS_CPU) / 4 , 1}, { NUM_THREADS_CPU / 2, 1},
	{ NUM_THREADS_CPU / 4, 1}, { NUM_THREADS_CPU / 8, 1}};
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS_ADDITIONAL_PARAMS{};
const std::array<unsigned int, 2> PARALLEL_PARAMS_DEFAULT{{NUM_THREADS_CPU, 1}};

//functions in RunAndEvaluateBpResults use above constants
#include "RunAndEvaluateBpResults.h"

template<typename T, unsigned int NUM_SET, bool TEMPLATED_DISP_IN_OPT_IMP>
void runBpOnSetAndUpdateResults(std::array<std::map<std::string, std::vector<std::string>>, 2>& resDefParParamsFinal) {
	std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>> runBpStereoSingleThread =
		std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
	if constexpr (TEMPLATED_DISP_IN_OPT_IMP) {
		std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>> optCPUDispValsInTemplate = 
			std::make_unique<RunBpStereoOptimizedCPU<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
		RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET>(resDefParParamsFinal, optCPUDispValsInTemplate, runBpStereoSingleThread);
	}
	else {
		std::unique_ptr<RunBpStereoSet<T, 0>> optCPUDispValsInTemplate = 
			std::make_unique<RunBpStereoOptimizedCPU<T, 0>>();
		RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET>(resDefParParamsFinal, optCPUDispValsInTemplate, runBpStereoSingleThread);
	}
}

int main(int argc, char** argv)
{
	std::array<std::map<std::string, std::vector<std::string>>, 2> resDefParParamsFinal;
	runBpOnSetAndUpdateResults<float, 0, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 0, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 1, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 1, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 2, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 2, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 3, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 3, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 4, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 4, false>(resDefParParamsFinal);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<float, 5, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 5, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 6, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float, 6, false>(resDefParParamsFinal);
#endif //SMALLER_SETS_ONLY
#ifdef DOUBLE_PRECISION_SUPPORTED
	runBpOnSetAndUpdateResults<double, 0, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 0, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 1, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 1, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 2, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 2, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 3, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 3, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 4, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 4, false>(resDefParParamsFinal);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<double, 5, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 5, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 6, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<double, 6, false>(resDefParParamsFinal);
#endif //SMALLER_SETS_ONLY
#endif //DOUBLE_PRECISION_SUPPORTED
#ifdef HALF_PRECISION_SUPPORTED
#ifdef COMPILING_FOR_ARM
	runBpOnSetAndUpdateResults<float16_t, 0, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 0, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 1, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 1, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 2, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 2, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 3, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 3, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 4, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 4, false>(resDefParParamsFinal);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<float16_t, 5, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 5, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 6, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<float16_t, 6, false>(resDefParParamsFinal);
#endif SMALLER_SETS_ONLY
#else
	runBpOnSetAndUpdateResults<short, 0, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 0, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 1, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 1, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 2, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 2, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 3, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 3, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 4, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 4, false>(resDefParParamsFinal);
#ifndef SMALLER_SETS_ONLY
	runBpOnSetAndUpdateResults<short, 5, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 5, false>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 6, true>(resDefParParamsFinal);
	runBpOnSetAndUpdateResults<short, 6, false>(resDefParParamsFinal);
#endif //SMALLER_SETS_ONLY
#endif //COMPILING_FOR_ARM
#endif //HALF_PRECISION_SUPPORTED

  	if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
		//retrieve and print average and median speedup using optimized
		//parallel parameters compared to default
		RunAndEvaluateBpResults::getAverageMedianSpeedup(resDefParParamsFinal);
	}

	//write results from default and optimized parallel parameters runs to csv file
	std::array<std::ofstream, 2> resultsStreamDefaultTBFinal{std::ofstream(BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE),
	                                                         std::ofstream(BP_ALL_RUNS_OUTPUT_CSV_FILE)};
	const auto headersInOrder = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).second;
	for (const auto& currHeader : headersInOrder) {
		resultsStreamDefaultTBFinal[0] << currHeader << ",";
		resultsStreamDefaultTBFinal[1] << currHeader << ",";
	}
	resultsStreamDefaultTBFinal[1] << "Speedup Over Default Parallel Parameters" << ",";

	resultsStreamDefaultTBFinal[0] << std::endl;
	resultsStreamDefaultTBFinal[1] << std::endl;

    for (unsigned int i=0; i < resultsStreamDefaultTBFinal.size(); i++) {
		for (unsigned int j=0; j < resDefParParamsFinal[i].begin()->second.size(); j++) {
			for (auto& currHeader : headersInOrder) {
				resultsStreamDefaultTBFinal[i] << resDefParParamsFinal[i].at(currHeader).at(j) << ",";
			}
			if (i == 1) {
				resultsStreamDefaultTBFinal[1] << resDefParParamsFinal[i].at("Speedup Over Default Parallel Parameters").at(j) << ",";
			}
			resultsStreamDefaultTBFinal[i] << std::endl;
		}
	}
	resultsStreamDefaultTBFinal[0].close();
	resultsStreamDefaultTBFinal[1].close();

	std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (optimized parallel parameters) in "
			  << BP_ALL_RUNS_OUTPUT_CSV_FILE << std::endl;
	std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (default parallel parameters) in "
			  << BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE << std::endl;
}
