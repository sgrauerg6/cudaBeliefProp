/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//Declares the methods to run Stereo BP on a series of images

#ifndef RUN_BP_STEREO_STEREO_SET_ON_GPU_WITH_CUDA_H
#define RUN_BP_STEREO_STEREO_SET_ON_GPU_WITH_CUDA_H

#include <cuda_runtime.h>
#include "BpConstsAndParams/bpStereoCudaParameters.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpRunProcessing/ProcessBPOnTargetDevice.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunImpCUDA/RunImpCUDAMemoryManagement.h"
#include "ProcessCUDABP.h"
#include "SmoothImageCUDA.h"

namespace bp_cuda_device
{
  inline RunData retrieveDeviceProperties(int numDevice)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, numDevice);
    int cudaDriverVersion;
    cudaDriverGetVersion(&cudaDriverVersion);
    int cudaRuntimeVersion;
    cudaRuntimeGetVersion(&cudaRuntimeVersion);

    RunData runData;
    runData.addDataWHeader("Device " + std::to_string(numDevice),
      std::string(prop.name) + " with " + std::to_string(prop.multiProcessorCount) + " multiprocessors");
    runData.addDataWHeader("Cuda version", std::to_string(cudaDriverVersion));
    runData.addDataWHeader("Cuda Runtime Version", std::to_string(cudaRuntimeVersion));
    return runData;
  }
};

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpStereoSetOnGPUWithCUDA : public RunBpStereoSet<T, DISP_VALS, ACCELERATION>
{
public:
  std::string getBpRunDescription() override { return "CUDA"; }

  //run the disparity map estimation BP on a set of stereo images and save the results between each set of images
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings, 
    const ParallelParams& parallelParams) override
  {
    //return null if acceleration setting is not CUDA
    if constexpr (ACCELERATION != run_environment::AccSetting::CUDA) {
      return {};
    }

    //using SmoothImageCUDA::SmoothImage;
    //generate struct with pointers to objects for running CUDA implementation and call
    //function to run CUDA implementation
    RunData runData;
    runData.addDataWHeader("CURRENT RUN", "GPU WITH CUDA");
    runData.appendData(bp_cuda_device::retrieveDeviceProperties(0));
    auto procSetOutput = this->processStereoSet(refTestImagePath, algSettings,
      BpOnDevice<T, DISP_VALS, ACCELERATION>{
        std::make_unique<SmoothImageCUDA>(parallelParams),
        std::make_unique<ProcessCUDABP<T, DISP_VALS, ACCELERATION>>(parallelParams),
        std::make_unique<RunImpCUDAMemoryManagement<T>>(),
        std::make_unique<RunImpCUDAMemoryManagement<float>>()});
    if (procSetOutput) {
      runData.appendData(procSetOutput->runData);
      procSetOutput->runData = runData;
    }
    
    return procSetOutput;
  }
};

//float16_t data type used for arm (rather than short)
//TODO: needs to be updated with other code changes
#ifdef COMPILING_FOR_ARM

template<>
class RunBpStereoSetOnGPUWithCUDA<float16_t, float16_t*> : public RunBpStereoSet<float16_t, float16_t*>
{
public:
  RunBpStereoSetOnGPUWithCUDA() {}

  std::string getBpRunDescription() override  { return "CUDA"; }

  //if type is specified as short, process as half on GPU
  //note that half is considered a data type for 16-bit floats in CUDA
  std::optional<ProcessStereoSetOutput> operator() (const std::string& refImagePath, const std::string& testImagePath,
    const beliefprop::BPsettings& algSettings, std::ostream& resultsStream, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<short>* runBpStereo = nullptr, RunImpMemoryManagement* memManagementImages = nullptr) override
  {

    //std::cout << "Processing as half on GPU\n";
    RunBpStereoSetOnGPUWithCUDA<halftype> runCUDABpStereoSet;
    ProcessCUDABP<halftype> runCUDABPHalfPrecision;
    return runCUDABpStereoSet(refImagePath,
      testImagePath,
      algSettings,
      saveDisparityMapImagePath,
      resultsStream,
      smoothImage,
      &runCUDABPHalfPrecision,
      memManagementImages);
  }
};

#endif //COMPILING_FOR_ARM

#ifdef _WIN32

extern "C" __declspec(dllexport) RunBpStereoSet<float, 0> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat();
extern "C" __declspec(dllexport) RunBpStereoSet<double, 0> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble();
extern "C" __declspec(dllexport) RunBpStereoSet<short, 0> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_> * __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp6();

#endif //_WIN32

#endif //RUN_BP_STEREO_IMAGE_SERIES_HEADER_CUH
