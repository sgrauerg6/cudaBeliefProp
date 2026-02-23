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
#include "ProcessBnchmrksDevice.h"

namespace benchmarks {

/**
 * @brief Structure with output runtime and other evaluation data from running
 * benchmarks implementation
 */
struct BnchmrksRunOutput
{
  std::chrono::duration<double> run_time;
  RunData run_data;
};

};

/**
 * @brief Abstract class to set up and run benchmarks on target device
 * using specified acceleration
 * 
 * @tparam T 
 * @tparam ACCELERATION 
 */
template<RunData_t T, run_environment::AccSetting ACCELERATION>
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
  virtual std::optional<benchmarks::BnchmrksRunOutput> operator()(
    unsigned int size,
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
  std::optional<benchmarks::BnchmrksRunOutput> ProcessBenchmarks(
    unsigned int size,
    const std::unique_ptr<ProcessBnchmrksDevice<T, ACCELERATION>>& proc_bnchmrks_device,
    const std::unique_ptr<MemoryManagement<T>>& mem_management) const;
};

//protected function to set up, run, and evaluate bp processing on target
//device using pointers to acceleration-specific smooth image,
//process BP, and memory management child class objects
template<RunData_t T, run_environment::AccSetting ACCELERATION>
std::optional<benchmarks::BnchmrksRunOutput> RunBnchmrks<T, ACCELERATION>::ProcessBenchmarks(
  unsigned int size,
  const std::unique_ptr<ProcessBnchmrksDevice<T, ACCELERATION>>& proc_bnchmrks_device,
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
  const std::size_t num_data_mat = size * size;
  
  auto start_time = std::chrono::system_clock::now();
  mat_0_device = mem_management->AllocateAlignedMemoryOnDevice(num_data_mat, ACCELERATION);
  mat_1_device = mem_management->AllocateAlignedMemoryOnDevice(num_data_mat, ACCELERATION);
  mat_2_device = mem_management->AllocateAlignedMemoryOnDevice(num_data_mat, ACCELERATION);
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> timeAllocate = end_time - start_time;

  //generate vector of random values of type T for mat_0_host and mat_1_host
  std::vector<T> mat_0_host(num_data_mat);
  std::vector<T> mat_1_host(num_data_mat);

  //allocate space on host for output matrix
  T* out_mat_host = new T[num_data_mat];
  
  start_time = std::chrono::system_clock::now();
  //Initialize a random number engine with a seed
  //Using steady_clock provides a non-deterministic seed
  unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
  std::mt19937 mersenne_engine(seed); //Mersenne Twister engine

  //Define the distribution to be values from -999 to 999
  std::uniform_real_distribution dist(-999.0, 999.0);

  //Use std::generate to fill the vector
  //A lambda function is used to bind the distribution and engine
  auto generator = [&]() { return dist(mersenne_engine); };
  std::generate(mat_0_host.begin(), mat_0_host.end(), generator);
  std::generate(mat_1_host.begin(), mat_1_host.end(), generator);
  end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> timeRandGenInMat = end_time - start_time;

  //transfer matrices from host to device
  auto start_time_run_w_transfer = std::chrono::system_clock::now();
  mem_management->TransferDataFromHostToDevice(
    mat_0_device,
    mat_0_host.data(),
    num_data_mat);
  mem_management->TransferDataFromHostToDevice(
    mat_1_device,
    mat_1_host.data(),
    num_data_mat);

  //run benchmark on device and retrieve output that includes
  //runtimes and other info about run
  auto start_time_run_no_transfer = std::chrono::system_clock::now();
  const auto process_bnchmrks_output = (*proc_bnchmrks_device)(
    size, mat_0_device, mat_1_device, mat_2_device);
  if (process_bnchmrks_output == run_eval::Status::kError) {
    return {};
  }
  auto end_time_run_no_transfer = std::chrono::system_clock::now();
  std::chrono::duration<double> runtimeNoTransfer =
    end_time_run_no_transfer - start_time_run_no_transfer;

  //transfer output data from device to host
  mem_management->TransferDataFromDeviceToHost(
    out_mat_host,
    mat_2_device,
    num_data_mat);
  auto end_time_run_w_transfer = std::chrono::system_clock::now();
  std::chrono::duration<double> runtimeWTransfer =
    end_time_run_w_transfer - start_time_run_w_transfer;

  //free aligned memory on device
  mem_management->FreeAlignedMemoryOnDevice(mat_0_device);
  mem_management->FreeAlignedMemoryOnDevice(mat_1_device);
  mem_management->FreeAlignedMemoryOnDevice(mat_2_device);

  //print output matrix
  for (size_t i=0; i < num_data_mat; i++) {
    std::cout << out_mat_host[i] << " ";
  }

  //add runtime data to data to return with corresponding headers
  run_data.AddDataWHeader("Time to allocate data in device", timeAllocate.count());
  run_data.AddDataWHeader("Time to generate random matrices", timeRandGenInMat.count());
  run_data.AddDataWHeader("Total time (no transfer)", runtimeNoTransfer.count());
  run_data.AddDataWHeader("Total time (including transfer)", runtimeWTransfer.count());

  benchmarks::BnchmrksRunOutput run_bnchmrks_output;
  //runtime without transfer time is stored as output runtime
  //with more detailed breakdowns in the run data
  run_bnchmrks_output.run_time = runtimeNoTransfer;
  run_bnchmrks_output.run_data = run_data;

  //free output matrix on host
  delete [] out_mat_host;

  //return output runtime and other info
  return run_bnchmrks_output;
}

#endif //RUN_BNCHMRKS_H