/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"
#include <thread>

template <typename T>
RunBpStereoOptimizedCPU<T>::RunBpStereoOptimizedCPU() {
}

template <typename T>
RunBpStereoOptimizedCPU<T>::~RunBpStereoOptimizedCPU() {
}

template<typename T>
ProcessStereoSetOutput RunBpStereoOptimizedCPU<T>::operator()(const std::string& refImagePath, const std::string& testImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream)
{
	resultsStream << "CURRENT RUN: OPTIMIZED CPU\n";
	int nthreads = std::thread::hardware_concurrency();
	omp_set_num_threads(nthreads);

	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}

	resultsStream << "Number of OMP threads: " << nthreads << "\n";
	std::unique_ptr<SmoothImage<>> smoothImageCPU = std::make_unique<SmoothImageCPU<>>();
	std::unique_ptr<ProcessBPOnTargetDevice<T, T*>> processImageCPU = std::make_unique<ProcessOptimizedCPUBP<T, T*>>();

	//can use default memory management since running on CPU
	return this->processStereoSet(refImagePath, testImagePath, algSettings,
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

template class RunBpStereoOptimizedCPU<float>;
template class RunBpStereoOptimizedCPU<double>;
#ifdef COMPILING_FOR_ARM
template class RunBpStereoOptimizedCPU<float16_t>;
#else
template class RunBpStereoOptimizedCPU<short>;
#endif //COMPILING_FOR_ARM
