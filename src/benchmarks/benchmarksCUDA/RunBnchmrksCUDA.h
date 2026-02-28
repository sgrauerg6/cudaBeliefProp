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
 * @file RunBnchmrksCUDA.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_BNCHMRKS_CUDA_H
#define RUN_BNCHMRKS_CUDA_H

#include "benchmarksRunProcessing/RunBnchmrks.h"
#include "ProcessBnchmrksCUDA.h"

class RunBnchmrksCUDA : public RunBnchmrks {
public:
  std::string RunDescription() const override { return std::string(run_cuda::kCUDADesc); }

  //run the benchmark(s) using the optimized CPU implementation
  std::optional<benchmarks::BnchmrksRunOutput<T>> operator()(
    const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
    const ParallelParams& parallel_params) const override;
};

template<RunData_t T, run_environment::AccSetting ACCELERATION>
inline std::optional<benchmarks::BnchmrksRunOutput<T>> RunBnchmrksCUDA<T, ACCELERATION>::operator()(
  const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
  const ParallelParams& parallel_params) const
{
  //return no value if acceleration setting is not CUDA
  if constexpr (ACCELERATION != run_environment::AccSetting::kCUDA) {
    return {};
  }

  //generate struct with pointers to objects for running CUDA implementation and call
  //function to run CUDA implementation
  RunData run_data;
  run_data.AppendData(run_cuda::retrieveDeviceProperties(0));
  auto process_bnchmrks_output = this->ProcessBenchmarks(
    inMtrces,
    std::make_unique<ProcessBnchmrksCUDA<T, ACCELERATION>>(parallel_params),
    std::make_unique<MemoryManagement<T>>());
  if (!process_bnchmrks_output) {
    return {};
  }

  run_data.AppendData(std::move(process_bnchmrks_output->run_data));
  process_bnchmrks_output->run_data = std::move(run_data);
    
  return process_bnchmrks_output;
}

#endif //RUN_BNCHMRKS_CUDA_H