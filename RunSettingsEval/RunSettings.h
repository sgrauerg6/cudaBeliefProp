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
#include <ranges>
#include "RunData.h"

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

//constants for headers corresponding to run settings
constexpr std::string_view kCPUThreadsPinnedHeader{"CPU Threads Pinned To Socket"};
constexpr std::string_view kOmpPlacesHeader{"OMP_PLACES"};
constexpr std::string_view kOmpProcBindHeader{"OMP_PROC_BIND"};
constexpr std::string_view kNumCPUThreadsHeader{"Total number of CPU threads"};
constexpr std::string_view kBytesAlignMemHeader{"BYTES_ALIGN_MEMORY"};
constexpr std::string_view kNumDataAlignWidthHeader{"NUM_DATA_ALIGN_WIDTH"};

//class to adjust and retrieve settings corresponding to CPU threads pinned to socket
class CPUThreadsPinnedToSocket {
public:
  //adjust setting to specify that CPU threads to be pinned to socket or not
  //if true, set CPU threads to be pinned to socket via OMP_PLACES and OMP_PROC_BIND envionmental variable settings
  //if false, set OMP_PLACES and OMP_PROC_BIND environment variables to be blank
  //TODO: currently commented out since it doesn't seem to have any effect
  void operator()(bool cpu_threads_pinned) const {
    /*if (cpu_threads_pinned) {
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
  RunData SettingsAsRunData() const {
    RunData pinned_threads_settings;
    const std::string omp_places_setting = (std::getenv("OMP_PLACES") == nullptr) ? "" : std::getenv("OMP_PLACES");
    const std::string omp_proc_bind_setting = (std::getenv("OMP_PROC_BIND") == nullptr) ? "" : std::getenv("OMP_PROC_BIND");
    const bool cpu_threads_pinned = ((omp_places_setting == "sockets") && (omp_proc_bind_setting == "true"));
    pinned_threads_settings.AddDataWHeader(std::string(kCPUThreadsPinnedHeader), cpu_threads_pinned);
    pinned_threads_settings.AddDataWHeader(std::string(kOmpPlacesHeader), omp_places_setting);
    pinned_threads_settings.AddDataWHeader(std::string(kOmpProcBindHeader), omp_proc_bind_setting);
    return pinned_threads_settings;
  }
};

//mapping from data size to data type string
const std::map<std::size_t, std::string_view> kDataSizeToNameMap{
  {sizeof(float), "FLOAT"},
  {sizeof(double), "DOUBLE"},
  {sizeof(short), "HALF"}
};

//enum for acceleration setting
enum class AccSetting {
  kNone, kAVX256, kAVX512, kNEON, kCUDA
};

//get string corresponding to acceleration method at compile time
template <AccSetting ACCELERATION_SETTING>
constexpr std::string_view AccelerationString() {
  if constexpr (ACCELERATION_SETTING == AccSetting::kNEON) { return "NEON"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX256) { return "AVX256"; }
  else if constexpr (ACCELERATION_SETTING == AccSetting::kAVX512) { return "AVX512"; }
  else { return "DEFAULT"; }
}

//get string corresponding to acceleration method at run time
inline std::string_view AccelerationString(AccSetting acceleration_setting) {
  if (acceleration_setting == AccSetting::kNEON) { return AccelerationString<AccSetting::kNEON>(); }
  else if (acceleration_setting == AccSetting::kAVX256) { return AccelerationString<AccSetting::kAVX256>(); }
  else if (acceleration_setting == AccSetting::kAVX512) { return AccelerationString<AccSetting::kAVX512>(); }
  else { return "DEFAULT"; }
}

inline unsigned int GetBytesAlignMemory(AccSetting accel_setting) {
  //avx512 requires data to be aligned on 64 bytes
  return (accel_setting == AccSetting::kAVX512) ? 64 : 16;
}

inline unsigned int GetNumDataAlignWidth(AccSetting accel_setting) {
  //align width with 16 data values in AVX512
  return (accel_setting == AccSetting::kAVX512) ? 16 : 8;
}

//generate RunData object that contains description header with corresponding value for each run setting
template <AccSetting ACCELERATION_SETTING>
inline RunData RunSettings()  {
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(std::string(kNumCPUThreadsHeader), std::thread::hardware_concurrency());
  curr_run_data.AddDataWHeader(std::string(kBytesAlignMemHeader), GetBytesAlignMemory(ACCELERATION_SETTING));
  curr_run_data.AddDataWHeader(std::string(kNumDataAlignWidthHeader), GetNumDataAlignWidth(ACCELERATION_SETTING));
  curr_run_data.AppendData(CPUThreadsPinnedToSocket().SettingsAsRunData());
  return curr_run_data;
}

//enum that specifies whether or not to use templated counts for the number of iterations in processing
//loops or to run implementation with and without templated iteration counts
//templated counts for number of loop iterations can help with optimization but requires that the number of
//iterations be known at compile time
enum class TemplatedItersSetting {
  kRunOnlyTempated,
  kRunOnlyNonTemplated,
  kRunTemplatedAndNotTemplated
};

//enum for parallel parameters settings in run
enum class ParallelParamsSetting { kDefault, kOptimized };

//enum to specify if optimizing parallel parameters per kernel or using same parallel parameters across all kernels in run
//in initial testing optimizing per kernel is faster on GPU and using same parallel parameters across all kernels is faster
//on CPU
enum class OptParallelParamsSetting {
  kSameParallelParamsAllKernels,
  kAllowDiffKernelParallelParamsInRun
};

//structure that stores settings for current implementation run
struct RunImpSettings {
  TemplatedItersSetting templated_iters_setting;
  std::pair<bool, OptParallelParamsSetting> opt_parallel_params_setting;
  std::pair<std::array<unsigned int, 2>, std::vector<std::array<unsigned int, 2>>> p_params_default_opt_settings;
  std::optional<std::string> run_name;
  //path to baseline runtimes for optimized and single thread runs and template setting used to generate baseline runtimes
  std::optional<std::array<std::string_view, 2>> baseline_runtimes_path_desc;
  std::vector<std::pair<std::string, std::vector<unsigned int>>> subset_str_indices;

  //remove parallel parameters with less than specified number of threads
  void RemoveParallelParamBelowMinThreads(unsigned int min_threads) {
    const auto [first_remove, last_remove] = std::ranges::remove_if(p_params_default_opt_settings.second,
      [min_threads](const auto& p_params) { return p_params[0] < min_threads; });
    p_params_default_opt_settings.second.erase(first_remove, last_remove);
  }

  //remove parallel parameters with greater than specified number of threads
  void RemoveParallelParamAboveMaxThreads(unsigned int max_threads) {
    const auto [first_remove, last_remove] = std::ranges::remove_if(p_params_default_opt_settings.second,
      [max_threads](const auto& p_params) { return p_params[0] > max_threads; });
    p_params_default_opt_settings.second.erase(first_remove, last_remove);
  }
};

};

#endif /* RUNSETTINGS_H_ */
