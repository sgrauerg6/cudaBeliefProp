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
 * @file RunBenchmarksOptCPU.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_BENCHMARKS_OPT_CPU_H
#define RUN_BENCHMARKS_OPT_CPU_H

#include "benchmarksRunProcessing/RunBenchmarks.h"

template<RunData_t T, run_environment::AccSetting ACCELERATION>
class RunBenchmarksOptCPU : public RunBenchmarks<T, ACCELERATION> {
public:
  std::string BnchmarksRunDescription() const override { return std::string(run_cpu::kBpOptCPUDesc); }

  //run the benchmark(s) using the optimized CPU implementation
  std::optional<benchmarks::BnchmrksRunOutput> operator()(
    unsigned int size,
    const ParallelParams& parallel_params) const override;
};

template<RunData_t T, run_environment::AccSetting ACCELERATION>
inline std::optional<benchmarks::BnchmrksRunOutput> RunBpImpOptCPU<T, DISP_VALS, ACCELERATION>::operator()(
  unsigned int size,
  const ParallelParams& parallel_params) const
{
    //set number of threads to use when running code in parallel using OpenMP from input parallel parameters
  //current setting on CPU is to execute all parallel processing in a run using the same number of parallel threads
  const unsigned int nthreads = 
    parallel_params.OptParamsForKernel({static_cast<size_t>(beliefprop::BpKernel::kBlurImages), 0})[0];
  #ifndef __APPLE__
    omp_set_num_threads(nthreads);
  #endif //__APPLE__

  //add settings for current run to output data
  RunData run_data;
  run_data.AddDataWHeader(
    std::string(run_cpu::kNumParallelCPUThreadsHeader),
    nthreads);
  run_data.AddDataWHeader(
    std::string(run_cpu::kCPUVectorizationHeader),
    std::string(run_environment::AccelerationString<ACCELERATION>()));

  //generate struct with pointers to objects for running optimized CPU implementation and call
  //function to run optimized CPU implementation
  auto process_set_output = this->ProcessBenchmarks(
    size,
    std::make_unique<ProcessBenchmarks<T, ACCELERATION>>(parallel_params),
    std::make_unique<MemoryManagement<T>>());
  if (process_set_output) {
    run_data.AppendData(std::move(process_set_output->run_data));
    process_set_output->run_data = std::move(run_data);
  }

  return process_set_output;
}

};

#endif //RUN_BENCHMARKS_OPT_CPU_H