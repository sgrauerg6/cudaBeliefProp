#pragma once

#define NOMINMAX
#include <memory>
#include <fstream>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include <map>

#include "BpConstsAndParams/RunSettings.h"
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpRunProcessing/RunBpStereoSet.h"

//typedef RunBpStereoSet<float>* (__cdecl *RunBpStereoSet_factory)();
template <typename T, unsigned int NUM_DISP>
using RunBpStereoSet_factory = RunBpStereoSet<T, NUM_DISP>* (__cdecl*)();

namespace run_bp_dlls
{
  enum class device_run { SINGLE_THREAD_CPU, OPTIMIZED_CPU, CUDA };
  const std::map<device_run, std::string> DLL_FILE_NAMES = { {device_run::SINGLE_THREAD_CPU, "BpSingleThreadCPUBeliefPropDLL.dll"},
                                {device_run::OPTIMIZED_CPU, "OptimizedCPUBeliefPropDLL.dll"},
                                {device_run::CUDA, "CUDABeliefPropDLL.dll"} };
}

//template <typename T= RunBpStereoSet<float, 16>* (__cdecl*)()>
class RunBpWithDLLsHelpers
{
public:
  static std::wstring s2ws(const std::string& s)
  {
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
  }

  //retrieve name of factory function to get run stereo set class object to run stereo set for each possible device config
  //templated with data type to process since function name is different for each data type (currently support float, double, 
  //and short) and also index of supported compile-time disparity (compTimeDispIdx set to <0 for case where disparity not known at compile-time)
  template <typename T>
  static const std::map<run_bp_dlls::device_run, std::string> getRunStereoSetMethodNames(const int compTimeDispIdx)
  {
    if (compTimeDispIdx < 0) {
      return std::map< run_bp_dlls::device_run, std::string> { {run_bp_dlls::device_run::SINGLE_THREAD_CPU, "createRunBpStereoCPUSingleThreadFloat"},
      { run_bp_dlls::device_run::OPTIMIZED_CPU, "createRunBpStereoOptimizedCPUFloat" },
      { run_bp_dlls::device_run::CUDA, "createRunBpStereoSetOnGPUWithCUDAFloat" } };
    }
    else {
      return std::map< run_bp_dlls::device_run, std::string> { {run_bp_dlls::device_run::SINGLE_THREAD_CPU, std::string("createRunBpStereoCPUSingleThreadFloat") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx)},
      { run_bp_dlls::device_run::OPTIMIZED_CPU, std::string("createRunBpStereoOptimizedCPUFloat") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx) },
      { run_bp_dlls::device_run::CUDA, std::string("createRunBpStereoSetOnGPUWithCUDAFloat") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx) } };
    }
  }

  template <>
  static const std::map<run_bp_dlls::device_run, std::string> getRunStereoSetMethodNames<double>(const int compTimeDispIdx)
  {
    if (compTimeDispIdx < 0) {
      return std::map< run_bp_dlls::device_run, std::string> { {run_bp_dlls::device_run::SINGLE_THREAD_CPU, "createRunBpStereoCPUSingleThreadDouble"},
      { run_bp_dlls::device_run::OPTIMIZED_CPU, "createRunBpStereoOptimizedCPUDouble" },
      { run_bp_dlls::device_run::CUDA, "createRunBpStereoSetOnGPUWithCUDADouble" } };
    }
    else {
      return std::map< run_bp_dlls::device_run, std::string> { {run_bp_dlls::device_run::SINGLE_THREAD_CPU, std::string("createRunBpStereoCPUSingleThreadDouble") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx)},
      { run_bp_dlls::device_run::OPTIMIZED_CPU, std::string("createRunBpStereoOptimizedCPUDouble") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx) },
      { run_bp_dlls::device_run::CUDA, std::string("createRunBpStereoSetOnGPUWithCUDADouble") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx) } };
    }
  }

  template <>
  static const std::map<run_bp_dlls::device_run, std::string> getRunStereoSetMethodNames<short>(const int compTimeDispIdx)
  {
    if (compTimeDispIdx < 0) {
      return std::map< run_bp_dlls::device_run, std::string> { {run_bp_dlls::device_run::SINGLE_THREAD_CPU, "createRunBpStereoCPUSingleThreadShort"},
      { run_bp_dlls::device_run::OPTIMIZED_CPU, "createRunBpStereoOptimizedCPUShort" },
      { run_bp_dlls::device_run::CUDA, "createRunBpStereoSetOnGPUWithCUDAShort" } };
    }
    else {
      return std::map< run_bp_dlls::device_run, std::string> { {run_bp_dlls::device_run::SINGLE_THREAD_CPU, std::string("createRunBpStereoCPUSingleThreadShort") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx)},
      { run_bp_dlls::device_run::OPTIMIZED_CPU, std::string("createRunBpStereoOptimizedCPUShort") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx) },
      { run_bp_dlls::device_run::CUDA, std::string("createRunBpStereoSetOnGPUWithCUDAShort") + std::string("_KnownDisp") + std::to_string(compTimeDispIdx) } };
    }
  }

  //retrieve mapping of each possible device config to factory function to run stereo set with config
  template <typename T, unsigned int NUM_DISP>
  static const std::map<run_bp_dlls::device_run, RunBpStereoSet_factory<T, NUM_DISP>> getRunBpFactoryFuncts(const int compTimeDispIdx)
  {
    std::map<run_bp_dlls::device_run, RunBpStereoSet_factory<T, NUM_DISP>> runBpFactoryFuncts;
    std::map<run_bp_dlls::device_run, std::string> runStereoSetMethodNames = getRunStereoSetMethodNames<T>(compTimeDispIdx);

    //retrieve the factory functions for each possible device run
    for (const auto& deviceTypeAndDLL : run_bp_dlls::DLL_FILE_NAMES)
    {
      HINSTANCE dll_handle = ::LoadLibrary((s2ws(deviceTypeAndDLL.second)).c_str());
      if (!dll_handle) {
        std::cout << "Unable to load DLL!\n";
        continue;
      }

      // Get the function from the DLL
      std::cout << "Name: " << runStereoSetMethodNames[deviceTypeAndDLL.first] << std::endl;
      RunBpStereoSet_factory<T, NUM_DISP> factory_func = reinterpret_cast<RunBpStereoSet_factory<T, NUM_DISP>>(
        ::GetProcAddress(dll_handle, (runStereoSetMethodNames[deviceTypeAndDLL.first]).c_str()));
      if (!factory_func) {
        std::cout << "Unable to load factory_func from DLL!\n";
        ::FreeLibrary(dll_handle);
        continue;
      }
      runBpFactoryFuncts[deviceTypeAndDLL.first] = factory_func;
    }

    return runBpFactoryFuncts;
  }
};
