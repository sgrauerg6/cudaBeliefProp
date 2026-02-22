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
 * @file RunBnchmrksSingThreadCPU.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_BNCHMRKS_SING_THREAD_CPU_H_
#define RUN_BNCHMRKS_SING_THREAD_CPU_H_

#include "../benchmarksRunProcessing/RunBnchmrks.h"

/**
 * @brief Child class of RunBpImp to run single-threaded CPU implementation of benchmarks on a
 * given stereo set as defined by reference and test image file paths
 * 
 * @tparam T
 * @tparam ACCELERATION
 */
template<typename T, run_environment::AccSetting ACCELERATION>
class RunBnchmrksSingThreadCPU : public RunBnchmrks<T, ACCELERATION> {
public:
  std::optional<beliefprop::BpRunOutput> operator()(
    unsigned int size,
    const ParallelParams& parallel_params) const override;
  std::string RunDescription() const override { return "Single-Thread CPU"; }
};

template<RunData_t T, run_environment::AccSetting ACCELERATION>
inline std::optional<benchmarks::BnchmrksRunOutput> RunBnchmrksSingThreadCPU<T, ACCELERATION>::operator()(
  unsigned int size,
  const ParallelParams& parallel_params) const
{
  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  const auto process_set_output = this->ProcessBenchmarks(
    size,
    std::make_unique<ProcessBnchmrks<T, ACCELERATION>>(parallel_params),
    std::make_unique<MemoryManagement<T>>());
  if (process_set_output) {
    //clear all returned run data and add only the runtime since that is all that
    //is used in the single threaded implementation
    process_set_output->run_data.ClearData();
    process_set_output->run_data.AddDataWHeader(std::string(run_eval::kSingleThreadRuntimeHeader), runtime.count());
    run_data.AppendData(std::move(process_set_output->run_data));
    process_set_output->run_data = std::move(run_data);
  }

  return process_set_output;
}

#endif //RUN_BNCHMRKS_SING_THREAD_CPU_H_