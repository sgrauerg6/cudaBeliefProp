// BeliefPropVSUseDLLs.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX
#include <memory>
#include <fstream>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>

//needed for consts and functions for running bp using DLLs
#include "GetDllFuncts/RunBpWithDLLsHelpers.h"

//needed to run the implementation a stereo set using CUDA
#include "MainDriverFiles/RunAndEvaluateBpResults.h"

int main(int argc, char** argv)
{
	//get mapping of device config to factory function to retrieve run stereo set object for device config
	auto runBpFactoryFuncts = RunBpWithDLLsHelpers::getRunBpFactoryFuncts();

	//std::ofstream resultsStream("output.txt", std::ofstream::out);
	std::ostream resultsStream(std::cout.rdbuf());
	RunAndEvaluateBpResults::printParameters(resultsStream);

	//run single thread CPU and CUDA implementations and compare
	std::array<std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>, 2> bpProcessingImps = {
				std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>(runBpFactoryFuncts[run_bp_dlls::device_run::SINGLE_THREAD_CPU]()),
				std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>(runBpFactoryFuncts[run_bp_dlls::device_run::CUDA]())};
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare(resultsStream, bpProcessingImps);

	//run single thread CPU and optimized implementations and compare
	std::array<std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>, 2> bpProcessingImps2 = {
				std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>(runBpFactoryFuncts[run_bp_dlls::device_run::SINGLE_THREAD_CPU]()),
				std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>(runBpFactoryFuncts[run_bp_dlls::device_run::OPTIMIZED_CPU]())};
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare(resultsStream, bpProcessingImps2);
}
