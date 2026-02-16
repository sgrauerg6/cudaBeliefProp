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
 * @file RunBenchmarks.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_BENCHMARKS_H
#define RUN_BENCHMARKS_H

namespace benchmarks {

/**
 * @brief Structure with output runtime and other evaluation data
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
class RunBenchmarks {
  /**
   * @brief Pure virtual function to return run description corresponding to
   * target acceleration
   * 
   * @return Description of run using specified acceleration
   */
  virtual std::string BpRunDescription() const = 0;

  /**
   * @brief Virtual destructor
   */
  virtual ~RunBenchmarks() {}

  /**
   * @brief Pure virtual operator() that must be defined in child class
   * 
   * @param ref_test_image_path 
   * @param alg_settings 
   * @param parallel_params 
   * @return Output from benchmarks run or null output if error
   */
  virtual std::optional<beliefprop::BnchmrksRunOutput> operator()(
    unsigned int size,
    const MemoryManagement<T>* mem_management, 
    const ParallelParams& parallel_params) const = 0;

protected:
  /**
   * @brief Protected function to set up, run, and evaluate benchmarks on
   * target device
   * 
   * @param mem_management memory management for device 
   * @return Output from running benchmarks or null if
   * error in run
   */
  std::optional<benchmarks::BpRunOutput> ProcessBenchmarks(
    unsigned int size,
    const MemoryManagement<T>* mem_management) const;
};

//protected function to set up, run, and evaluate bp processing on target
//device using pointers to acceleration-specific smooth image,
//process BP, and memory management child class objects
template<RunData_t T, run_environment::AccSetting ACCELERATION>
std::optional<benchmarks::BpRunOutput> RunBenchmarks<T, DISP_VALS, ACCELERATION>::ProcessBenchmarks(
  unsigned int size,
  const MemoryManagement<T>* mem_management) const
{
//allocate data for bp processing on target device ahead of runs if option selected
  T* bp_data{nullptr};
  T* bp_proc_store{nullptr};
  if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
    //allocate memory on device for bp processing
    const std::size_t num_data =
      BpLevel<T>::TotalDataForAlignedMemoryAllLevels(
        width_height_images,
        alg_settings.num_disp_vals,
        alg_settings.num_levels,
        ACCELERATION);
    bp_data = 
      run_bp_on_device.mem_management_bp_run->AllocateAlignedMemoryOnDevice(
        10*num_data,
        ACCELERATION);
    if (run_bp_on_device.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) !=
        run_eval::Status::kNoError) { return {}; }

    BpLevel<T> bottom_bp_level(width_height_images, 0, 0, ACCELERATION);
    const std::size_t total_data_bottom_level =
      bottom_bp_level.NumDataInBpArrays(alg_settings.num_disp_vals);
    bp_proc_store =
      run_bp_on_device.mem_management_bp_run->AllocateAlignedMemoryOnDevice(
        total_data_bottom_level,
        ACCELERATION);
    if (run_bp_on_device.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) !=
        run_eval::Status::kNoError) { return {}; }
  }
}

#endif //RUN_BENCHMARKS_H