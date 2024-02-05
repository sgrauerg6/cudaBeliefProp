/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float, 0>* __cdecl createRunBpStereoOptimizedCPUFloat() {
  return new RunBpStereoOptimizedCPU<float, 0>();
}

__declspec(dllexport) RunBpStereoSet<double, 0>* __cdecl createRunBpStereoOptimizedCPUDouble() {
  return new RunBpStereoOptimizedCPU<double, 0>();
}

__declspec(dllexport) RunBpStereoSet<short, 0>* __cdecl createRunBpStereoOptimizedCPUShort() {
  return new RunBpStereoOptimizedCPU<short, 0>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp0() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp0() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp0() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp1() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp1() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp1() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp2() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp2() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp2() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp3() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp3() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp3() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp4() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp4() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp4() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp5() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp5() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp5() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp6() {
  return new RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp6() {
  return new RunBpStereoOptimizedCPU<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(BpParallelParams(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp6() {
  return new RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(BpParallelParams(bp_params::LEVELS_BP));
}

#endif //_WIN32
