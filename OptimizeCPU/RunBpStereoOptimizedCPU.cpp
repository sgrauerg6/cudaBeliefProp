/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"
#include <thread>

template <typename T, unsigned int DISP_VALS>
RunBpStereoOptimizedCPU<T, DISP_VALS>::RunBpStereoOptimizedCPU() { }

template <typename T, unsigned int DISP_VALS>
RunBpStereoOptimizedCPU<T, DISP_VALS>::~RunBpStereoOptimizedCPU() { }

template<typename T, unsigned int DISP_VALS>
ProcessStereoSetOutput RunBpStereoOptimizedCPU<T, DISP_VALS>::operator()(const std::array<std::string, 2>& refTestImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream)
{
	resultsStream << "CURRENT RUN: OPTIMIZED CPU\n";
	unsigned int nthreads = std::thread::hardware_concurrency();
	omp_set_num_threads(nthreads);

	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}

	resultsStream << "Number of OMP threads: " << nthreads << "\n";
	std::unique_ptr<SmoothImage<>> smoothImageCPU = std::make_unique<SmoothImageCPU<>>();
	std::unique_ptr<ProcessBPOnTargetDevice<T, T*, DISP_VALS>> processImageCPU =
			std::make_unique<ProcessOptimizedCPUBP<T, T*, DISP_VALS>>();

	//can use default memory management since running on CPU
	return this->processStereoSet(refTestImagePath, algSettings,
			resultsStream, smoothImageCPU, processImageCPU);
}

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float>* __cdecl createRunBpStereoOptimizedCPUFloat()
{
	return new RunBpStereoOptimizedCPU<float>();
}

__declspec(dllexport) RunBpStereoSet<double>* __cdecl createRunBpStereoOptimizedCPUDouble()
{
	return new RunBpStereoOptimizedCPU<double>();
}

__declspec(dllexport) RunBpStereoSet<short>* __cdecl createRunBpStereoOptimizedCPUShort()
{
	return new RunBpStereoOptimizedCPU<short>();
}

#endif //_WIN32

template class RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;
template class RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;
#ifdef COMPILING_FOR_ARM
template class RunBpStereoOptimizedCPU<float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;
#else
template class RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;
#endif //COMPILING_FOR_ARM
