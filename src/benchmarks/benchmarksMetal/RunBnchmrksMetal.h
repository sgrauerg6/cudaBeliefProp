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
 * @file RunBnchmrksMetal.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_BNCHMRKS_METAL_H
#define RUN_BNCHMRKS_METAL_H

#include "benchmarksRunProcessing/RunBnchmrks.h"
#include "ProcessBnchmrksMetal.h"
#include "RunImpMetal/RunMetalSettings.h"
#include "RunImpMetal/MemoryManagementMetal.h"
#include <Metal/Metal.hpp>

template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
class RunBnchmrksMetal : public RunBnchmrks<T, ACCELERATION, BENCHMARK_RUN> {
public:
  std::string RunDescription() const override { return std::string(run_metal::kMetalDesc); }

  //run the benchmark(s) using the optimized CPU implementation
  std::optional<benchmarks::BnchmrksRunOutput<T>> operator()(
    const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
    const ParallelParams& parallel_params) const override;
};

template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
inline std::optional<benchmarks::BnchmrksRunOutput<T>> RunBnchmrksMetal<T, ACCELERATION, BENCHMARK_RUN>::operator()(
  const std::array<BnchmrksMtrx<T>, 2>& inMtrces,
  const ParallelParams& parallel_params) const
{
  std::cout << "RUN METAL 1" << std::endl;
  //return no value if acceleration setting is not CUDA
  if constexpr (ACCELERATION != run_environment::AccSetting::kMETAL) {
    return {};
  }
  std::cout << "RUN METAL 2" << std::endl;

  //generate struct with pointers to objects for running CUDA implementation and call
  //function to run CUDA implementation
  RunData run_data;
  constexpr size_t kNumEvalRuns{7};
  auto m_device = MTL::CreateSystemDefaultDevice();
  auto process_bnchmrks_output = this->template ProcessBenchmarks<MTL::Buffer>(
    inMtrces,
    std::make_unique<ProcessBnchmrksMetal<T, ACCELERATION, BENCHMARK_RUN, MTL::Buffer>>(parallel_params, m_device),
    std::make_unique<MemoryManagementMetal<T, MTL::Buffer>>(m_device),
    kNumEvalRuns);
  if (!process_bnchmrks_output) {
    return {};
  }

  run_data.AppendData(std::move(process_bnchmrks_output->run_data));
  process_bnchmrks_output->run_data = std::move(run_data);
    
  return process_bnchmrks_output;
}

#endif //RUN_BNCHMRKS_METAL_H