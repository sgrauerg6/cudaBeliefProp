/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include <iostream>
#include <string>
#include <memory>
#include <array>
#include <thread>
#include "SmoothImageCPU.h"
#include "ProcessOptimizedCPUBP.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"

template <typename T, unsigned int DISP_VALS>
class RunBpStereoOptimizedCPU : public RunBpStereoSet<T, DISP_VALS> {
public:
	RunBpStereoOptimizedCPU() {}
	virtual ~RunBpStereoOptimizedCPU() {}

	std::string getBpRunDescription() override { return "Optimized CPU"; }

	//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
	ProcessStereoSetOutput operator()(const std::array<std::string, 2>& refTestImagePath, const BPsettings& algSettings, std::ostream& resultsStream) override;
};

template<typename T, unsigned int DISP_VALS>
inline ProcessStereoSetOutput RunBpStereoOptimizedCPU<T, DISP_VALS>::operator()(const std::array<std::string, 2>& refTestImagePath,
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

extern "C" __declspec(dllexport) RunBpStereoSet<float, 0>* __cdecl createRunBpStereoOptimizedCPUFloat();
extern "C" __declspec(dllexport) RunBpStereoSet<double, 0>* __cdecl createRunBpStereoOptimizedCPUDouble();
extern "C" __declspec(dllexport) RunBpStereoSet<short, 0>* __cdecl createRunBpStereoOptimizedCPUShort();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]> * __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]> * __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]> * __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]> * __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]> * __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp5();

#endif //_WIN32

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
