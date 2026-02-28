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
 * @file RunBnchmrks.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_BNCHMRKS_H
#define RUN_BNCHMRKS_H

#include <chrono>
#include <optional>
#include <vector>
#include <random>
#include <algorithm>
#include "RunSettingsParams/RunSettingsConstsEnums.h"
#include "RunEval/RunData.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/MemoryManagement.h"
#include "benchmarksResultsEval/DetailedTimingBnchmrksConsts.h"
#include "ProcessBnchmrksDevice.h"
#include "BnchmrksMtrx.h"
#include "BnchmrksConstsEnumsAliases.h"

namespace benchmarks {

/**
 * @brief Structure with output runtime and other evaluation data from running
 * benchmarks implementation as well as the output matrix from the run
 */
template<RunData_t T>
struct BnchmrksRunOutput
{
  std::chrono::duration<double> run_time;
  RunData run_data;
  BnchmrksMtrx<T> result_mtrx;
};

};

/**
 * @brief Abstract class to set up and run benchmarks on target device
 * using specified acceleration
 * 
 * @tparam T 
 * @tparam ACCELERATION 
 */
template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
class RunBnchmrks {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~RunBnchmrks() {}

  /**
   * @brief Pure virtual function to return run description corresponding to
   * target acceleration
   * 
   * @return Description of run using specified acceleration
   */
  virtual std::string RunDescription() const = 0;

  /**
   * @brief Pure virtual operator() that must be defined in child class
   * 
   * @param size 
   * @param parallel_params 
   * @return Output from benchmarks run or null output if error
   */
  virtual std::optional<benchmarks::BnchmrksRunOutput<T>> operator()(
    const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
    const ParallelParams& parallel_params) const = 0;

protected:
  /**
   * @brief Protected function to set up, run, and evaluate benchmarks on
   * target device
   * 
   * @param size
   * @param proc_bnchmrks_device
   * @param mem_management memory management for device 
   * @return Output from running benchmarks or null if
   * error in run
   */
  std::optional<benchmarks::BnchmrksRunOutput<T>> ProcessBenchmarks(
    const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
    const std::unique_ptr<ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN>>& proc_bnchmrks_device,
    const std::unique_ptr<MemoryManagement<T>>& mem_management) const;
};

//protected function to set up, run, and evaluate bp processing on target
//device using pointers to acceleration-specific smooth image,
//process BP, and memory management child class objects
template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
std::optional<benchmarks::BnchmrksRunOutput<T>> RunBnchmrks<T, ACCELERATION, BENCHMARK_RUN>::ProcessBenchmarks(
  const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
  const std::unique_ptr<ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN>>& proc_bnchmrks_device,
  const std::unique_ptr<MemoryManagement<T>>& mem_management) const
{
  //initialize run data to include timing data and possibly
  //other info
  RunData run_data;

  //allocate data for benchmark processing on target device
  T* mat_0_device{nullptr};
  T* mat_1_device{nullptr};
  T* mat_2_device{nullptr};

  //allocate memory on device for benchmark processing
  const std::size_t num_data_mat = inMtrces[0].Width() * inMtrces[0].Height();
  
  auto start_time = std::chrono::system_clock::now();
  mat_0_device = mem_management->AllocateAlignedMemoryOnDevice(num_data_mat, ACCELERATION);
  mat_1_device = mem_management->AllocateAlignedMemoryOnDevice(num_data_mat, ACCELERATION);
  mat_2_device = mem_management->AllocateAlignedMemoryOnDevice(num_data_mat, ACCELERATION);
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> timeAllocate = end_time - start_time;

  //allocate space on host for output matrix
  T* out_mat_host = new T[num_data_mat];

  //transfer matrices from host to device
  mem_management->TransferDataFromHostToDevice(
    mat_0_device,
    inMtrces[0].Data(),
    num_data_mat);
  mem_management->TransferDataFromHostToDevice(
    mat_1_device,
    inMtrces[1].Data(),
    num_data_mat);

  //initialize structures for timing data
  DetailedTimings detailed_bnchmrks_timings(benchmarks::kTimingNames);

  //run benchmark(s) on device a specified number of times and use median
  //runtime(s) across runs in results
  constexpr size_t kNumEvalRuns{3};
  for (size_t i=0; i < kNumEvalRuns; i++) {
    //run benchmark(s) on device and retrieve output runtime(s)
    const auto process_bnchmrks_timings = (*proc_bnchmrks_device)(
      inMtrces[0].Width(), mat_0_device, mat_1_device, mat_2_device);
    if (!process_bnchmrks_timings) {
      return {};
    }
    //add timing(s) from current run to overall timings
    detailed_bnchmrks_timings.AddToCurrentTimings(*process_bnchmrks_timings);
  }

  //transfer output data from device to host
  mem_management->TransferDataFromDeviceToHost(
    out_mat_host,
    mat_2_device,
    num_data_mat);

  //free aligned memory on device
  mem_management->FreeAlignedMemoryOnDevice(mat_0_device);
  mem_management->FreeAlignedMemoryOnDevice(mat_1_device);
  mem_management->FreeAlignedMemoryOnDevice(mat_2_device);

  //print random value from output matrix
  unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
  std::mt19937 mersenne_engine(seed); //Mersenne Twister engine
  std::cout << out_mat_host[
    std::uniform_int_distribution<int>(1, num_data_mat)(mersenne_engine)]
             << std::endl;

  //add runtime data to data to return with corresponding headers
  run_data.AddDataWHeader(
    std::string(run_eval::kNumEvalRuns),
    static_cast<unsigned int>(kNumEvalRuns));
  run_data.AddDataWHeader(
    std::string(benchmarks::kTimingNames.at(
      benchmarks::Runtime_Type::kAddMatNoTransfer)),
    detailed_bnchmrks_timings.MedianTiming(
      benchmarks::Runtime_Type::kAddMatNoTransfer).count());

  benchmarks::BnchmrksRunOutput<T> run_bnchmrks_output;
  //runtime without transfer time is stored as output runtime
  //with more detailed breakdowns in the run data
  run_bnchmrks_output.run_time =
    detailed_bnchmrks_timings.MedianTiming(
      benchmarks::Runtime_Type::kAddMatNoTransfer);
  run_bnchmrks_output.run_data = run_data;
  //copy results matrix to output
  run_bnchmrks_output.result_mtrx = BnchmrksMtrx<T>(
    inMtrces[0].Width(),
    inMtrces[0].Height(),
    out_mat_host);

  //free output matrix on host
  delete [] out_mat_host;

  //return output runtime and other info
  return run_bnchmrks_output;
}

#endif //RUN_BNCHMRKS_H