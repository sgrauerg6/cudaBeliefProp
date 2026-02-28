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
 * @file ProcessBnchmrksOptCPU.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BNCHMRKS_OPT_CPU_H_
#define PROCESS_BNCHMRKS_OPT_CPU_H_

#include <chrono>
#include "benchmarksRunProcessing/ProcessBnchmrksDevice.h"
#include "KernelBnchmrksOptCPU.h"

template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
class ProcessBnchmrksOptCPU final : public ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN> {
public:
/**
 * @brief Constructor to initialize class to process benchmarks
 * in optimized CPU implementation
 * 
 * @param opt_cpu_params Parallel parameters to use in implementation
 */
explicit ProcessBnchmrksOptCPU(const ParallelParams& opt_cpu_params) : 
    ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN>(opt_cpu_params) {}

private:
  /**
   * @brief Function to run add matrices benchmark on device
   * 
   * @param mat_w_h
   * @param mat_addend_0
   * @param mat_addend_1
   * @param mat_sum
   * @return Status of "no error" if successful, "error" status otherwise
   */
  std::optional<DetailedTimings<benchmarks::Runtime_Type>> AddMatrices(
    const unsigned int mat_w_h,
    const T* mat_addend_0,
    const T* mat_addend_1,
    T* mat_sum) const override
  {
    auto add_mat_start_time = std::chrono::system_clock::now();
    if constexpr (ACCELERATION == run_environment::AccSetting::kNone) {
      benchmarks_cpu::AddMatricesNoPackedInstructions<T>(
        mat_w_h, mat_w_h, mat_addend_0, mat_addend_1, mat_sum);
    }
#if defined(COMPILING_FOR_ARM)
    else if constexpr (ACCELERATION == run_environment::AccSetting::kNEON) {
      std::cout << "Processing NEON implementation" << std::endl;
      benchmarks_cpu::AddMatricesUseSIMDVectorsNEON(
        mat_w_h, mat_w_h, mat_addend_0, mat_addend_1, mat_sum);
    }
#else
    else if constexpr (
      (ACCELERATION == run_environment::AccSetting::kAVX256) ||
      (ACCELERATION == run_environment::AccSetting::kAVX256_F16))
    {
      std::cout << "Processing AVX256 implementation" << std::endl;
      benchmarks_cpu::AddMatricesUseSIMDVectorsAVX256(
        mat_w_h, mat_w_h, mat_addend_0, mat_addend_1, mat_sum);
    }
    else if constexpr (
      (ACCELERATION == run_environment::AccSetting::kAVX512) ||
      (ACCELERATION == run_environment::AccSetting::kAVX512_F16))
    {
      std::cout << "Processing AVX512 implementation" << std::endl;
      benchmarks_cpu::AddMatricesUseSIMDVectorsAVX512(
        mat_w_h, mat_w_h, mat_addend_0, mat_addend_1, mat_sum);
    }
#endif //COMPILING_FOR_ARM
    auto end_mat_start_time = std::chrono::system_clock::now();
    DetailedTimings add_mat_timing(benchmarks::kTimingNames);
    add_mat_timing.AddTiming(benchmarks::Runtime_Type::kAddMatNoTransfer,
      end_mat_start_time - add_mat_start_time);
    return add_mat_timing;
  }
};

#endif //PROCESS_BNCHMRKS_OPT_CPU_H_