/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file ProcessBenchmarks.h
 * @author Scott Grauer-Gray
 * @brief Declares abstract class to run benchmarks on target device.
 * Some of the class functions need to be overridden for processing on
 * target device.
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BENCHMARKS_H_
#define PROCESS_BENCHMARKS_H_

#include <math.h>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <ranges>
#include "RunSettingsParams/RunSettings.h"
#include "RunImp/MemoryManagement.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpResultsEvaluation/DetailedTimingBpConsts.h"
#include "BpSettings.h"
#include "BpRunUtils.h"
#include "ParallelParamsBp.h"
#include "BpLevel.h"

/** @brief Alias for time point for start and end time for each timing
 *  segment */
using timingType = std::chrono::time_point<std::chrono::system_clock>;

/**
 * @brief Abstract class to run belief propagation on target device.
 * Some of the class functions need to be overridden for processing on
 * target device.
 * 
 * @tparam T 
 * @tparam ACCELERATION 
 */
template<RunData_t T, run_environment::AccSetting ACCELERATION>
class ProcessBenchmarks {
public:
  explicit ProcessBenchmarks(
    const ParallelParams& parallel_params) :
    parallel_params_{parallel_params} {}
  
  /**
   * @brief Virtual destructor
   */
  virtual ~ProcessBenchmarks() {}

  virtual run_eval::Status ErrorCheck(
    const char *file = "",
    int line = 0,
    bool abort = false) const
  {
    return run_eval::Status::kNoError;
  }
  
  /**
   * @brief Run benchmarks implementation.<br>
   * Input is benchmark size<br>
   * Output is runtimes
   * 
   * @param benchmark_size 
   * @param alg_settings 
   * @param width_height_images 
   * @param allocated_mem_bp_processing 
   * @param allocated_memory 
   * @param mem_management
   * @return std::optional<std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>> 
   */
  std::optional<std::pair<float*, DetailedTimings<beliefprop::Runtime_Type>>> operator()(
    const unsigned int benchmark_size,
    T* mat_0_allocated_memory,
    T* mat_1_allocated_memory,
    T* mat_2_allocated_memory,
    const MemoryManagement<T>* mem_management) const;

protected:
  const ParallelParams& parallel_params_;

private:
  /**
   * @brief Pure virtual function to run add matrices benchmark on device<br>
   * Must be defined in child class corresponding to device
   * 
   * @param mat_0
   * @param mat_1
   * @param mat_2
   * @param mat_w_h
   * @return Status of "no error" if successful, "error" status otherwise
   */
  virtual run_eval::Status AddMatrices(
    const T* mat_0,
    const T* mat_1,
    const T* mat_2,
    const unsigned int mat_w_h) const = 0;
};

#endif //PROCESS_BENCHMARKS_H_