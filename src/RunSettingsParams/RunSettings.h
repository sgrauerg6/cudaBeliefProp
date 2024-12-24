/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file RunSettings.h
 * @author Scott Grauer-Gray
 * @brief Declares and defines structure that stores settings for current
 * implementation run as well as functions related to run settings
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUNSETTINGS_H_
#define RUNSETTINGS_H_

#include <optional>
#include <string_view>
#include <thread>
#include <ranges>
#include <set>
#include "RunEval/RunData.h"
#include "InputSignature.h"
#include "RunSettingsConstsEnums.h"
#include "CPUThreadsPinnedToSocket.h"

namespace run_environment {

inline unsigned int GetBytesAlignMemory(AccSetting accel_setting) {
  //avx512 requires data to be aligned on 64 bytes
  return ((accel_setting == AccSetting::kAVX512) ||
          (accel_setting == AccSetting::kAVX512_F16)) ?
          64 :
          32;
}

/**
 * @brief Generate RunData object that contains description header with
 * corresponding value for each run setting
 * 
 * @tparam ACCELERATION_SETTING 
 * @return RunData 
 */
template <AccSetting ACCELERATION_SETTING>
inline RunData RunSettings()  {
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(std::string(kNumCPUThreadsHeader), std::thread::hardware_concurrency());
  curr_run_data.AddDataWHeader(std::string(kBytesAlignMemHeader), GetBytesAlignMemory(ACCELERATION_SETTING));
  curr_run_data.AppendData(CPUThreadsPinnedToSocket().SettingsAsRunData());
  return curr_run_data;
}

/**
 * @brief Structure that stores settings for current implementation run
 */
struct RunImpSettings {
  std::vector<unsigned int> datatypes_eval_sizes;
  TemplatedItersSetting templated_iters_setting;
  OptParallelParamsSetting opt_parallel_params_setting;
  std::pair<std::array<unsigned int, 2>, std::set<std::array<unsigned int, 2>>> p_params_default_alt_options;
  std::optional<std::string> run_name;
  //path to baseline runtimes for optimized and single thread runs and template setting used to generate baseline runtimes
  std::optional<std::array<std::string_view, 2>> baseline_runtimes_path_desc;
  std::vector<std::pair<std::string, std::vector<InputSignature>>> subset_desc_input_sig;
  bool run_alt_optimized_imps;

  /**
   * @brief Remove parallel parameters with less than specified number of threads
   * 
   * @param min_threads 
   */
  void RemoveParallelParamBelowMinThreads(unsigned int min_threads) {
    std::erase_if(p_params_default_alt_options.second,
      [min_threads](const auto& p_params) { return p_params[0] < min_threads; });
  }

  /**
   * @brief Remove parallel parameters with greater than specified number of threads
   * 
   * @param max_threads 
   */
  void RemoveParallelParamAboveMaxThreads(unsigned int max_threads) {
    std::erase_if(p_params_default_alt_options.second,
      [max_threads](const auto& p_params) { return p_params[0] > max_threads; });
  }
};

};

#endif /* RUNSETTINGS_H_ */
