/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float>* __cdecl createRunBpStereoOptimizedCPUFloat() {
	return new RunBpStereoOptimizedCPU<float>();
}

__declspec(dllexport) RunBpStereoSet<double>* __cdecl createRunBpStereoOptimizedCPUDouble() {
	return new RunBpStereoOptimizedCPU<double>();
}

__declspec(dllexport) RunBpStereoSet<short>* __cdecl createRunBpStereoOptimizedCPUShort() {
	return new RunBpStereoOptimizedCPU<short>();
}

#endif //_WIN32
