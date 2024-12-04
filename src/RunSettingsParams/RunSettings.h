/*
 * RunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef RUNSETTINGS_H_
#define RUNSETTINGS_H_

#include <optional>
#include <string_view>
#include <thread>
#include <ranges>
#include "RunEval/RunData.h"
#include "InputSignature.h"
#include "RunSettingsConstsEnums.h"
#include "CPUThreadsPinnedToSocket.h"

namespace run_environment {

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

//structure that stores settings for current implementation run
struct RunImpSettings {
  TemplatedItersSetting templated_iters_setting;
  std::pair<bool, OptParallelParamsSetting> opt_parallel_params_setting;
  std::pair<std::array<unsigned int, 2>, std::vector<std::array<unsigned int, 2>>> p_params_default_opt_settings;
  std::optional<std::string> run_name;
  //path to baseline runtimes for optimized and single thread runs and template setting used to generate baseline runtimes
  std::optional<std::array<std::string_view, 2>> baseline_runtimes_path_desc;
  std::vector<std::pair<std::string, std::vector<InputSignature>>> subset_desc_input_sig;

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
