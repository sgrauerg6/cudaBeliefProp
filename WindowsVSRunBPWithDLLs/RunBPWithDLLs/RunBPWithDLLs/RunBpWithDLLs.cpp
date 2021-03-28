// BeliefPropVSUseDLLs.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX
#include <memory>
#include <fstream>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include "ParameterFiles/bpStereoParameters.h"

//needed for consts and functions for running bp using DLLs
#include "GetDllFuncts/RunBpWithDLLsHelpers.h"

//needed to run the implementation a stereo set using CUDA
#include "MainDriverFiles/RunAndEvaluateBpResults.h"

int main(int argc, char** argv)
{
	//get mapping of device config to factory function to retrieve run stereo set object for device config
	auto runBpFactoryFuncts = RunBpWithDLLsHelpers::getRunBpFactoryFuncts(0);

	//set bp settings for image processing
	BPsettings algSettings;
	algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0];

	//std::ofstream resultsStream("output.txt", std::ofstream::out);
	std::ostream resultsStream(std::cout.rdbuf());
	RunAndEvaluateBpResults::printParameters(0, resultsStream);

	//run single thread CPU and CUDA implementations and compare
	std::array<std::unique_ptr<RunBpStereoSet<float, 16>>, 2> bpProcessingImps = {
				std::unique_ptr<RunBpStereoSet<float, 16>>(runBpFactoryFuncts[run_bp_dlls::device_run::SINGLE_THREAD_CPU]()),
				std::unique_ptr<RunBpStereoSet<float, 16>>(runBpFactoryFuncts[run_bp_dlls::device_run::CUDA]())};
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, 16>(resultsStream, bpProcessingImps, 0, algSettings);

	//run single thread CPU and optimized implementations and compare
	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>, 2> bpProcessingImps2 = {
				std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>(runBpFactoryFuncts[run_bp_dlls::device_run::SINGLE_THREAD_CPU]()),
				std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>(runBpFactoryFuncts[run_bp_dlls::device_run::OPTIMIZED_CPU]())};
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(resultsStream, bpProcessingImps2, 0, algSettings);
}
