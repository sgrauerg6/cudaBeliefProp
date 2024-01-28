/*
 * RunBpStereoOptimizedCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOOPTIMIZEDCPU_H_
#define RUNBPSTEREOOPTIMIZEDCPU_H_

#include <string>
#include <memory>
#include <array>
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "SmoothImageCPU.h"
#include "ProcessOptimizedCPUBP.h"

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
class RunBpStereoOptimizedCPU : public RunBpStereoSet<T, DISP_VALS, VECTORIZATION> {
public:
  RunBpStereoOptimizedCPU() {}

  std::string getBpRunDescription() override { return "Optimized CPU"; }

  //run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
  ProcessStereoSetOutput operator()(const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings,
    const beliefprop::ParallelParameters& parallelParams) override;
};

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting VECTORIZATION>
inline ProcessStereoSetOutput RunBpStereoOptimizedCPU<T, DISP_VALS, VECTORIZATION>::operator()(const std::array<std::string, 2>& refTestImagePath,
  const beliefprop::BPsettings& algSettings, const beliefprop::ParallelParameters& parallelParams)
{
  unsigned int nthreads = parallelParams.parallelDimsEachKernel_[beliefprop::BLUR_IMAGES][0][0];
  omp_set_num_threads(nthreads);

  RunData runData;
  runData.addDataWHeader("CURRENT RUN", "OPTIMIZED CPU");
  runData.addDataWHeader("Number of threads", std::to_string(nthreads));
  runData.addDataWHeader("Vectorization", run_environment::accelerationString<VECTORIZATION>());

  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  auto procSetOutput = this->processStereoSet(refTestImagePath, algSettings, 
    BpOnDevice<T, DISP_VALS, VECTORIZATION>{
      std::make_unique<SmoothImageCPU>(parallelParams),
      std::make_unique<ProcessOptimizedCPUBP<T, DISP_VALS, VECTORIZATION>>(parallelParams),
      std::make_unique<RunImpMemoryManagement<T>>(),
      std::make_unique<RunImpMemoryManagement<float>>()});
  runData.appendData(procSetOutput.runData);
  procSetOutput.runData = runData;

  return procSetOutput;
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
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]> * __cdecl createRunBpStereoOptimizedCPUFloat_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]> * __cdecl createRunBpStereoOptimizedCPUDouble_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]> * __cdecl createRunBpStereoOptimizedCPUShort_KnownDisp6();

#endif //_WIN32

#endif /* RUNBPSTEREOOPTIMIZEDCPU_H_ */
