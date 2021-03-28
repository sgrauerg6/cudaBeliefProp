/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUFloat() {
	return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUDouble() {
	return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUShort() {
	return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>();
}

#endif //_WIN32
