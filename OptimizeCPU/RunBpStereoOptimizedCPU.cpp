/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"
#include "ProcessOptimizedCPUBP.h"

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
	int nthreads = 0;

	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}

	resultsStream << "Number of OMP threads: " << nthreads << "\n";
	std::unique_ptr<SmoothImage> smoothImageCPU = std::make_unique<SmoothImageCPU>();
	std::unique_ptr<ProcessBPOnTargetDevice<T>> processImageCPU = std::make_unique<ProcessOptimizedCPUBP<T>>();
	return this->processStereoSet(refImagePath, testImagePath, algSettings,
			resultsStream, smoothImageCPU, processImageCPU);
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class RunBpStereoOptimizedCPU<float>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class RunBpStereoOptimizedCPU<double>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class RunBpStereoOptimizedCPU<float16_t>;
#else
template class RunBpStereoOptimizedCPU<short>;
#endif //COMPILING_FOR_ARM

#endif
