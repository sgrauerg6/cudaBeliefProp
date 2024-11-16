/*
 * RunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef RUNSETTINGS_H_
#define RUNSETTINGS_H_

#include <map>
#include <optional>
#include <string_view>
#include <thread>
#include <iostream>
#include <algorithm>
#include "RunSettingsEval/RunData.h"

//set data type used for half-precision
#ifdef OPTIMIZED_CPU_RUN
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
using halftype = float16_t;
#else
using halftype = short;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CPU_RUN

namespace run_environment {

//class to adjust and retrieve settings corresponding to CPU threads pinned to socket
class CPUThreadsPinnedToSocket {
public:
  //adjust setting to specify that CPU threads to be pinned to socket or not
  //if true, set CPU threads to be pinned to socket via OMP_PLACES and OMP_PROC_BIND envionmental variable settings
  //if false, set OMP_PLACES and OMP_PROC_BIND environment variables to be blank
  //TODO: currently commented out since it doesn't seem to have any effect
  void operator()(bool cpuThreadsPinned) const {
    /*if (cpuThreadsPinned) {
      int success = system("export OMP_PLACES=\"sockets\"");
      if (success == 0) {
        std::cout << "export OMP_PLACES=\"sockets\" success" << std::endl;
      }
      success = system("export OMP_PROC_BIND=true");
      if (success == 0) {
        std::cout << "export OMP_PROC_BIND=true success" << std::endl;
      }
    }
    else {
      int success = system("export OMP_PLACES=");
      if (success == 0) {
        std::cout << "export OMP_PLACES= success" << std::endl;
      }
      success = system("export OMP_PROC_BIND=");
      if (success == 0) {
        std::cout << "export OMP_PROC_BIND= success" << std::endl;
      }
    }*/
  }

  //retrieve environment variable values corresponding to CPU threads being pinned to socket and return
  //as RunData structure
  RunData currSettingsAsRunData() const {
    RunData pinnedThreadsSettings;
    const std::string ompPlacesSetting = (std::getenv("OMP_PLACES") == nullptr) ? "" : std::getenv("OMP_PLACES");
    const std::string ompProcBindSetting = (std::getenv("OMP_PROC_BIND") == nullptr) ? "" : std::getenv("OMP_PROC_BIND");
    const bool cpuThreadsPinned = ((ompPlacesSetting == "sockets") && (ompProcBindSetting == "true"));
    pinnedThreadsSettings.addDataWHeader("CPU Threads Pinned To Socket", cpuThreadsPinned);
    pinnedThreadsSettings.addDataWHeader("OMP_PLACES", ompPlacesSetting);
    pinnedThreadsSettings.addDataWHeader("OMP_PROC_BIND", ompProcBindSetting);
    return pinnedThreadsSettings;
  }
};

//mapping from data size to data type string
const std::map<std::size_t, std::string_view> DATA_SIZE_TO_NAME_MAP{
  {sizeof(float), "FLOAT"},
  {sizeof(double), "DOUBLE"},
  {sizeof(short), "HALF"}
};

//enum for acceleration setting
enum class AccSetting {
  NONE, AVX256, AVX512, NEON, CUDA
};

//get string corresponding to acceleration method at compile time
template <AccSetting ACCELERATION_SETTING>
constexpr std::string_view accelerationString() {
  if constexpr (ACCELERATION_SETTING == AccSetting::NEON) { return "NEON"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::AVX256) { return "AVX256"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::AVX512) { return "AVX512"; }
  else { return "DEFAULT"; }
}

//get string corresponding to acceleration method at run time
inline std::string_view accelerationString(AccSetting accelerationSetting) {
  if (accelerationSetting == AccSetting::NEON) { return accelerationString<AccSetting::NEON>(); }
  else if (accelerationSetting == AccSetting::AVX256) { return accelerationString<AccSetting::AVX256>(); }
  else if (accelerationSetting == AccSetting::AVX512) { return accelerationString<AccSetting::AVX512>(); }
  else { return "DEFAULT"; }
}

inline unsigned int getBytesAlignMemory(AccSetting accelSetting) {
  //avx512 requires data to be aligned on 64 bytes
  return (accelSetting == AccSetting::AVX512) ? 64 : 16;
}

inline unsigned int getNumDataAlignWidth(AccSetting accelSetting) {
  //align width with 16 data values in AVX512
  return (accelSetting == AccSetting::AVX512) ? 16 : 8;
}

//generate RunData object that contains description header with corresponding value for each run setting
template <AccSetting ACCELERATION_SETTING>
inline RunData runSettings()  {
  RunData currRunData;
  currRunData.addDataWHeader("Total number of CPU threads", std::thread::hardware_concurrency());
  currRunData.addDataWHeader("BYTES_ALIGN_MEMORY", getBytesAlignMemory(ACCELERATION_SETTING));
  currRunData.addDataWHeader("NUM_DATA_ALIGN_WIDTH", getNumDataAlignWidth(ACCELERATION_SETTING));
  currRunData.appendData(CPUThreadsPinnedToSocket().currSettingsAsRunData());
  return currRunData;
}

//enum that specifies whether or not to use templated counts for the number of iterations in processing
//loops or to run implementation with and without templated iteration counts
//templated counts for number of loop iterations can help with optimization but requires that the number of
//iterations be known at compile time
enum class TemplatedItersSetting {
  RUN_ONLY_TEMPLATED,
  RUN_ONLY_NON_TEMPLATED,
  RUN_TEMPLATED_AND_NOT_TEMPLATED
};

//enum to specify if optimizing parallel parameters per kernel or using same parallel parameters across all kernels in run
//in initial testing optimizing per kernel is faster on GPU and using same parallel parameters across all kernels is faster
//on CPU
enum class OptParallelParamsSetting {
  SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN,
  ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN
};

//structure that stores settings for current implementation run
struct RunImpSettings {
  TemplatedItersSetting templatedItersSetting_;
  std::pair<bool, OptParallelParamsSetting> optParallelParamsOptionSetting_;
  std::pair<std::array<unsigned int, 2>, std::vector<std::array<unsigned int, 2>>> pParamsDefaultOptOptions_;
  std::optional<std::string> runName_;
  //path to baseline runtimes for optimized and single thread runs and template setting used to generate baseline runtimes
  std::optional<std::pair<std::array<std::string_view, 2>, TemplatedItersSetting>> baseOptSingThreadRTimeForTSetting_;
  std::vector<std::pair<std::string, std::vector<unsigned int>>> subsetStrIndices_;

  //remove parallel parameters with less than specified number of threads
  void removeParallelParamBelowMinThreads(const unsigned int minThreads) {
    pParamsDefaultOptOptions_.second.erase(
      std::remove_if(pParamsDefaultOptOptions_.second.begin(), pParamsDefaultOptOptions_.second.end(),
        [minThreads](const auto& pParams) { return pParams[0] < minThreads; }),
        pParamsDefaultOptOptions_.second.end());
  }

  //remove parallel parameters with greater than specified number of threads
  void removeParallelParamAboveMaxThreads(const unsigned int maxThreads) {
    pParamsDefaultOptOptions_.second.erase(
      std::remove_if(pParamsDefaultOptOptions_.second.begin(), pParamsDefaultOptOptions_.second.end(),
        [maxThreads](const auto& pParams) { return pParams[0] > maxThreads; }),
        pParamsDefaultOptOptions_.second.end());
  }
};

};

#endif /* RUNSETTINGS_H_ */
