/*
Copyright (C) 2024 Scott Grauer-Gray

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
 * @file RunBpOnStereoSet.h
 * @author Scott Grauer-Gray
 * @brief Declares abstract class to set up and run belief propagation on target device using
 * specified acceleration
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include <array>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <ranges>
#include "RunImp/MemoryManagement.h"
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunData.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "BpImageProcessing/BpImage.h"
#include "BpImageProcessing/DisparityMap.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpResultsEvaluation/DetailedTimingBpConsts.h"
#include "ParallelParamsBp.h"
#include "ProcessBp.h"

namespace beliefprop {

/**
 * @brief Structure with output disparity map, runtime, and other evaluation data
 */
struct BpRunOutput
{
  std::chrono::duration<double> run_time;
  DisparityMap<float> out_disparity_map;
  RunData run_data;
};

/**
 * @brief Structure with pointers to objects containing functions for smoothing images,
 * processing bp, and memory management on target device using specified acceleration
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
struct BpOnDevice {
  const std::unique_ptr<SmoothImage>& smooth_image;
  const std::unique_ptr<ProcessBp<T, DISP_VALS, ACCELERATION>>& run_bp_stereo;
  const std::unique_ptr<MemoryManagement<T>>& mem_management_bp_run;
  const std::unique_ptr<MemoryManagement<float>>& mem_management_images;
};

};

/**
 * @brief Abstract class to set up and run belief propagation on target device using
 * specified acceleration
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpOnStereoSet {
public:
  /**
   * @brief Pure virtual function to return run description corresponding to target acceleration
   * 
   * @return std::string 
   */
  virtual std::string BpRunDescription() const = 0;

  /**
   * @brief Pure virtual operator() overload that must be defined in child class
   * 
   * @param ref_test_image_path 
   * @param alg_settings 
   * @param parallel_params 
   * @return std::optional<beliefprop::BpRunOutput> 
   */
  virtual std::optional<beliefprop::BpRunOutput> operator()(
    const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings, 
    const ParallelParams& parallel_params) const = 0;

protected:
  /**
   * @brief Protected function to set up, run, and evaluate bp processing on
   * target device using pointers to acceleration-specific smooth image,
   * process BP, and memory management child class objects
   * 
   * @param ref_test_image_path 
   * @param alg_settings 
   * @param run_bp_on_device 
   * @return std::optional<beliefprop::BpRunOutput> 
   */
  std::optional<beliefprop::BpRunOutput> ProcessStereoSet(
    const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings,
    const beliefprop::BpOnDevice<T, DISP_VALS, ACCELERATION>& run_bp_on_device) const;
};

//protected function to set up, run, and evaluate bp processing on target device using pointers to acceleration-specific smooth image,
//process BP, and memory management child class objects
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
std::optional<beliefprop::BpRunOutput> RunBpOnStereoSet<T, DISP_VALS, ACCELERATION>::ProcessStereoSet(
  const std::array<std::string, 2>& ref_test_image_path,
  const beliefprop::BpSettings& alg_settings,
  const beliefprop::BpOnDevice<T, DISP_VALS, ACCELERATION>& run_bp_on_device) const
{
  std::cout << "2" << std::endl;
  //retrieve the images as well as the width and height
  const std::array<BpImage<unsigned int>, 2> input_images{
    BpImage<unsigned int>(ref_test_image_path[0]),
    BpImage<unsigned int>(ref_test_image_path[1])};
  const std::array<unsigned int, 2> width_height_images{input_images[0].Width(), input_images[0].Height()};

  //get total number of pixels in input images
  const unsigned int tot_num_pixels_images{width_height_images[0] * width_height_images[1]};

  //initialize structures for timing data
  std::unordered_map<beliefprop::Runtime_Type, std::array<timingType, 2>> runtime_start_end_timings;
  DetailedTimings detailed_bp_timings(beliefprop::kTimingNames);

  //initialize output disparity map
  DisparityMap<float> output_disparity_map(width_height_images);
  std::cout << "3" << std::endl;

  //allocate data for bp processing on target device ahead of runs if option selected
  T* bp_data{nullptr};
  T* bp_proc_store{nullptr};
  if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
    //allocate memory on device for bp processing
    const std::size_t num_data = BpLevel::TotalDataForAlignedMemoryAllLevels<T, ACCELERATION>(
      width_height_images, alg_settings.num_disp_vals, alg_settings.num_levels);
    bp_data = run_bp_on_device.mem_management_bp_run->AllocateAlignedMemoryOnDevice(10*num_data, ACCELERATION);
    if (run_bp_on_device.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) { return {}; }

    BpLevel bottom_bp_level(width_height_images, 0, 0, ACCELERATION);
    const std::size_t total_data_bottom_level = bottom_bp_level.NumDataInBpArrays<T>(alg_settings.num_disp_vals);
    bp_proc_store = run_bp_on_device.mem_management_bp_run->AllocateAlignedMemoryOnDevice(total_data_bottom_level, ACCELERATION);
    if (run_bp_on_device.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) { return {}; }
  }

  std::cout << "4" << std::endl;
  //run bp processing for specified number of runs
  const unsigned int num_evaluation_runs{beliefprop::NumBpStereoRuns(alg_settings.num_disp_vals)};
  for (unsigned int num_run = 0; 
       num_run < num_evaluation_runs;
       num_run++)
  {
    //allocate the device memory to store and x and y smoothed images
    std::array<float*, 2> smoothed_images{
      run_bp_on_device.mem_management_images->AllocateMemoryOnDevice(tot_num_pixels_images),
      run_bp_on_device.mem_management_images->AllocateMemoryOnDevice(tot_num_pixels_images)};
    
    if (run_bp_on_device.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      //return std::optional object with no value indicating error in run
      return {};
    }

    //set start timer for specified runtime segments at time before smoothing images
    runtime_start_end_timings[beliefprop::Runtime_Type::kSmoothing][0] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalNoTransfer][0] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalWithTransfer][0] = std::chrono::system_clock::now();

    //first smooth the images using the Gaussian filter with the given smoothing sigma value
    //smoothed images are stored on the target device
    for (unsigned int i = 0; i < 2; i++) {
      (*(run_bp_on_device.smooth_image))(input_images[i], alg_settings.smoothing_sigma, smoothed_images[i]);
      if (run_bp_on_device.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) { 
        //return std::optional object with no value indicating error in run
        return {};
      }
    }

    //end timer for image smoothing and add to image smoothing timings
    runtime_start_end_timings[beliefprop::Runtime_Type::kSmoothing][1] = std::chrono::system_clock::now();

    //get and store time point at start of bp processing
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalBp][0] = std::chrono::system_clock::now();

    //run belief propagation on device as specified by input pointer to ProcessBp object run_bp_stereo
    //returns detailed timings for bp run
    auto bp_stereo_output = (*(run_bp_on_device.run_bp_stereo))(
      smoothed_images, alg_settings, width_height_images,
      bp_data, bp_proc_store, run_bp_on_device.mem_management_bp_run);
    if (!bp_stereo_output) {
      //return std::optional object with no value if null output indicating error in run
      return {};
    }

    //get and store end timepoint of bp run for computation of total runtime
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalBp][1] =
      std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalNoTransfer][1] =
      std::chrono::system_clock::now();

    //transfer the disparity map estimation on the device to the host for output
    run_bp_on_device.mem_management_images->TransferDataFromDeviceToHost(
      output_disparity_map.PointerToPixelsStart(), bp_stereo_output->first, tot_num_pixels_images);

    //compute timings for each portion of interest and add to vector timings
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalWithTransfer][1] =
      std::chrono::system_clock::now();

    //retrieve the duration for each segment and add to bp timings
    std::ranges::for_each(runtime_start_end_timings,
      [&detailed_bp_timings] (const auto& current_runtime_name_timing) {
        detailed_bp_timings.AddTiming(current_runtime_name_timing.first,
          current_runtime_name_timing.second[1] - current_runtime_name_timing.second[0]);
    });

    //add bp timings from current run to overall timings
    detailed_bp_timings.AddToCurrentTimings(bp_stereo_output->second);

    //free the space allocated to the resulting disparity map and smoothed images on the computation device
    run_bp_on_device.mem_management_images->FreeMemoryOnDevice(bp_stereo_output->first);
    for (auto& smoothed_image : smoothed_images) {
      run_bp_on_device.mem_management_images->FreeMemoryOnDevice(smoothed_image);
    }
  }

  //free data for bp processing on target device if this memory
  //management set to be done outside of runs
  if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
    run_bp_on_device.mem_management_bp_run->FreeAlignedMemoryOnDevice(bp_data);
    run_bp_on_device.mem_management_bp_run->FreeAlignedMemoryOnDevice(bp_proc_store);
  }
  std::cout << "5" << std::endl;

  //construct RunData object with bp input and timing info
  RunData run_data;
  run_data.AddDataWHeader(std::string(beliefprop::kImageWidthHeader), width_height_images[0]);
  run_data.AddDataWHeader(std::string(beliefprop::kImageHeightHeader), width_height_images[1]);
  run_data.AddDataWHeader(std::string(beliefprop::kNumEvalRuns), num_evaluation_runs);
  run_data.AppendData(detailed_bp_timings.AsRunData());
  std::cout << "6" << std::endl;

  //construct and return beliefprop::BpRunOutput object inside of std::optional object
  return {beliefprop::BpRunOutput{
    detailed_bp_timings.MedianTiming(
      beliefprop::Runtime_Type::kTotalWithTransfer), std::move(output_disparity_map), run_data}};
}

#endif /* RUNBPSTEREOSET_H_ */
