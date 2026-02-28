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
 * @file ProcessBnchmrksDevice.h
 * @author Scott Grauer-Gray
 * @brief Declares abstract class to run benchmarks on target device.
 * Some of the class functions need to be overridden for processing on
 * target device.
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BNCHMRKS_DEVICE_H_
#define PROCESS_BNCHMRKS_DEVICE_H_

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
#include "benchmarksResultsEval/DetailedTimingBnchmrksConsts.h"
#include "benchmarksRunProcessing/BnchmrksConstsEnumsAliases.h"
#include "ParallelParamsBnchmrks.h"
#include "BnchmrksConstsEnumsAliases.h"

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
template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
class ProcessBnchmrksDevice {
public:
  explicit ProcessBnchmrksDevice(
    const ParallelParams& parallel_params) :
    parallel_params_{parallel_params} {}
  
  /**
   * @brief Virtual destructor
   */
  virtual ~ProcessBnchmrksDevice() {}

  virtual run_eval::Status ErrorCheck(
    const char *file = "",
    int line = 0,
    bool abort = false) const
  {
    return run_eval::Status::kNoError;
  }
  
  /**
   * @brief Run benchmarks implementation.<br>
   * Input is benchmark size, input matrices, and
   * allocated space for output matrix on device<br>
   * Output is runtimes
   * 
   * @param benchmark_size 
   * @param mat_input_0 
   * @param mat_input_1 
   * @param mat_output
   * @return std::optional<DetailedTimings<benchmarks::Runtime_Type>>
   */
  std::optional<DetailedTimings<benchmarks::Runtime_Type>> operator()(
    const unsigned int benchmark_size,
    const T* mat_input_0,
    const T* mat_input_1,
    T* mat_output) const
  {
    return AddMatrices(benchmark_size, mat_input_0, mat_input_1, mat_output);
  }

protected:
  const ParallelParams& parallel_params_;

private:
  /**
   * @brief Pure virtual function to run add matrices benchmark on device<br>
   * Must be defined in child class corresponding to device
   * 
   * @param mat_w_h
   * @param mat_addend_0
   * @param mat_addend_1
   * @param mat_sum
   * @return Status of "no error" if successful, "error" status otherwise
   */
  virtual std::optional<DetailedTimings<benchmarks::Runtime_Type>> AddMatrices(
    const unsigned int mat_w_h,
    const T* mat_addend_0,
    const T* mat_addend_1,
    T* mat_sum) const = 0;
};

#endif //PROCESS_BNCHMRKS_DEVICE_H_