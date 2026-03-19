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
 * @file ProcessBnchmrksHIP.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BNCHMRKS_HIP_H_
#define PROCESS_BNCHMRKS_HIP_H_

#include <hip_runtime.h>
#include <hip_fp16.h>
#include "benchmarksRunProcessing/ProcessBnchmrksDevice.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConsts.h"
#include "RunEval/RunEvalEnumsStructs.h"
#include "RunImpHIP/RunHIPSettings.h"

template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN, typename U = T>
class ProcessBnchmrksHIP : public ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN, U> {
public:
  /**
   * @brief Constructor to initialize class to process benchmarks
   * in HIP implementation
   * 
   * @param opt_cpu_params Parallel parameters to use in implementation
   */
  explicit ProcessBnchmrksHIP(const ParallelParams& opt_cpu_params) : 
    ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN>(opt_cpu_params) {}

private:
  /**
   * @brief Function to run add matrices benchmark on device
   * 
   * @param mat_w_h
   * @param mat_input_0
   * @param mat_input_1
   * @param mat_result
   * @return Status of "no error" if successful, "error" status otherwise
   */
  std::optional<DetailedTimings<benchmarks::Runtime_Type>> TwoDMatricesBnchmrk(
    const unsigned int mat_w_h,
    const U* mat_input_0,
    const U* mat_input_1,
    U* mat_result) const override;
};

#endif //PROCESS_BNCHMRKS_HIP_H_