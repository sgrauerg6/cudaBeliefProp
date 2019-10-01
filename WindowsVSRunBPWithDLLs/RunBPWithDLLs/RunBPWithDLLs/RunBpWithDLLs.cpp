// BeliefPropVSUseDLLs.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX
#include <memory>
#include <fstream>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>

//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "ParameterFiles/bpStereoCudaParameters.h"
#include "BpAndSmoothProcessing/RunBpStereoSet.h"
#include "FileProcessing/BpFileHandling.h"
#include "OutputEvaluation/DisparityMap.h"

//needed to run the implementation a stereo set using CUDA
#include "MainDriverFiles/RunAndEvaluateBpResults.h"

//typedef RunBpStereoSet<float>* (__cdecl *RunBpStereoSet_factory)();
using RunBpStereoSetFloat_factory = RunBpStereoSet<float>* (__cdecl*)();
using RunBpStereoSetDouble_factory = RunBpStereoSet<double>* (__cdecl*)();
using RunBpStereoSetShort_factory = RunBpStereoSet<short>* (__cdecl*)();

int main(int argc, char** argv)
{
	// Load the DLL
	HINSTANCE dll_handle = ::LoadLibrary(TEXT("CUDABeliefPropDLL.dll"));
	if (!dll_handle) {
		std::cout << "Unable to load DLL!\n";
		return 1;
	}

	// Get the function from the DLL
	RunBpStereoSetFloat_factory factory_funcCUDA = reinterpret_cast<RunBpStereoSetFloat_factory>(
		::GetProcAddress(dll_handle, "createRunBpStereoSetOnGPUWithCUDAFloat"));
	if (!factory_funcCUDA) {
		std::cout << "Unable to load createRunBpStereoSetOnGPUWithCUDAFloat from DLL!\n";
		::FreeLibrary(dll_handle);
		return 1;
	}

	// Load the DLL
	dll_handle = ::LoadLibrary(TEXT("SingleThreadCPUBeliefPropDLL.dll"));
	if (!dll_handle) {
		std::cout << "Unable to load DLL!\n";
		return 1;
	}

	// Get the function from the DLL
	RunBpStereoSetFloat_factory factory_funcSingleThreadCPU = reinterpret_cast<RunBpStereoSetFloat_factory>(
		::GetProcAddress(dll_handle, "createRunBpStereoCPUSingleThreadFloat"));
	if (!factory_funcSingleThreadCPU) {
		std::cout << "Unable to load createRunBpStereoCPUSingleThreadFloat from DLL!\n";
		::FreeLibrary(dll_handle);
		return 1;
	}

	// Load the DLL
	dll_handle = ::LoadLibrary(TEXT("OptimizedCPUBeliefPropDLL.dll"));
	if (!dll_handle) {
		std::cout << "Unable to load DLL!\n";
		return 1;
	}

	// Get the function from the DLL
	RunBpStereoSetFloat_factory factory_funcOptimizedCPU = reinterpret_cast<RunBpStereoSetFloat_factory>(
		::GetProcAddress(dll_handle, "createRunBpStereoOptimizedCPUFloat"));
	if (!factory_funcOptimizedCPU) {
		std::cout << "Unable to load createRunBpStereoOptimizedCPUFloat from DLL!\n";
		::FreeLibrary(dll_handle);
		return 1;
	}

	//std::ofstream resultsStream("output.txt", std::ofstream::out);
	std::ostream resultsStream(std::cout.rdbuf());
	RunAndEvaluateBpResults::printParameters(resultsStream);

	std::array<std::unique_ptr<RunBpStereoSet<float>>, 2> bpProcessingImps = {
				std::unique_ptr<RunBpStereoSet<float>>(factory_funcSingleThreadCPU()),
				std::unique_ptr<RunBpStereoSet<float>>(factory_funcCUDA())
	};

	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare(resultsStream, bpProcessingImps);
}
