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
 * @file ProcessBenchmarksCPU.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BENCHMARKS_CPU_H_
#define PROCESS_BENCHMARKS_CPU_H_

#include "benchmarksRunProcessing/ProcessBenchmarksDevice.h"
#include "KernelBenchmarksCPU.h"

template<RunData_t T, run_environment::AccSetting ACCELERATION>
class ProcessBenchmarksCPU : public ProcessBenchmarksDevice<T, ACCELERATION> {
public:

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
  run_eval::Status AddMatrices(
    const unsigned int mat_w_h,
    const T* mat_addend_0,
    const T* mat_addend_1,
    T* mat_sum) const override
  {
    benchmarks_cpu::AddMatricesNoPackedInstructions<T>(
      mat_w_h, mat_w_h, mat_addend_0, mat_addend_1, mat_sum);
    return run_eval::Status::kNoError;
  }
};

#endif //PROCESS_BENCHMARKS_CPU_H_