/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include <unordered_map>
#include <memory>
#include <array>
#include <string>
#include <optional>
#include <ranges>
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpResultsEvaluation/DetailedTimingBPConsts.h"
#include "BpImageProcessing/BpImage.h"
#include "BpImageProcessing/DisparityMap.h"
#include "BpImageProcessing/SmoothImage.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "RunImp/RunImpMemoryManagement.h"
#include "BpParallelParams.h"
#include "ProcessBPOnTargetDevice.h"

//structure with output disparity map, runtime, and other evaluation data
struct ProcessStereoSetOutput
{
  std::chrono::duration<double> run_time;
  DisparityMap<float> out_disparity_map;
  RunData run_data;
};

//structure with pointers to objects containing functions for smoothing images,
//processing bp, and memory management on target device using specified acceleration
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
struct BpOnDevice {
  const std::unique_ptr<SmoothImage>& smooth_image;
  const std::unique_ptr<ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>>& run_bp_stereo;
  const std::unique_ptr<RunImpMemoryManagement<T>>& mem_management_bp_run;
  const std::unique_ptr<RunImpMemoryManagement<float>>& mem_management_images;
};

//abstract class to set up and run belief propagation on target device using specified acceleration
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpStereoSet {
public:
  //pure virtual function to return run description corresponding to target acceleration
  virtual std::string BpRunDescription() const = 0;

  //pure virtual operator() overload that must be defined in child class
  virtual std::optional<ProcessStereoSetOutput> operator()(
    const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings, 
    const ParallelParams& parallel_params) const = 0;

protected:
  //protected function to set up, run, and evaluate bp processing on target device using pointers to acceleration-specific smooth image,
  //process BP, and memory management child class objects
  std::optional<ProcessStereoSetOutput> ProcessStereoSet(
    const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings,
    const BpOnDevice<T, DISP_VALS, ACCELERATION>& runBpOnDevice) const;
};

//protected function to set up, run, and evaluate bp processing on target device using pointers to acceleration-specific smooth image,
//process BP, and memory management child class objects
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
std::optional<ProcessStereoSetOutput> RunBpStereoSet<T, DISP_VALS, ACCELERATION>::ProcessStereoSet(
  const std::array<std::string, 2>& ref_test_image_path,
  const beliefprop::BpSettings& alg_settings,
  const BpOnDevice<T, DISP_VALS, ACCELERATION>& runBpOnDevice) const
{
  //retrieve the images as well as the width and height
  const std::array<BpImage<unsigned int>, 2> inputImages{
    BpImage<unsigned int>(ref_test_image_path[0]),
    BpImage<unsigned int>(ref_test_image_path[1])};
  const std::array<unsigned int, 2> width_height_images{inputImages[0].Width(), inputImages[0].Height()};

  //get total number of pixels in input images
  const unsigned int totNumPixelsImages{width_height_images[0] * width_height_images[1]};

  //initialize structures for timing data
  std::unordered_map<beliefprop::Runtime_Type, std::array<timingType, 2>> runtime_start_end_timings;
  DetailedTimings detailedBPTimings(beliefprop::kTimingNames);

  //initialize output disparity map
  DisparityMap<float> output_disparity_map(width_height_images);

  //allocate data for bp processing on target device ahead of runs if option selected
  T* bpData{nullptr};
  T* bpProcStore{nullptr};
  if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
    //allocate memory on device for bp processing
    const unsigned long numData = beliefprop::BpLevel::TotalDataForAlignedMemoryAllLevels<T, ACCELERATION>(
      width_height_images, alg_settings.num_disp_vals, alg_settings.num_levels);
    bpData = runBpOnDevice.mem_management_bp_run->AllocateAlignedMemoryOnDevice(10u*numData, ACCELERATION);
    if (runBpOnDevice.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) { return {}; }

    beliefprop::BpLevel bottomBpLevel(width_height_images, 0, 0, ACCELERATION);
    const unsigned long total_dataBottomLevel = bottomBpLevel.NumDataInBpArrays<T>(alg_settings.num_disp_vals);
    bpProcStore = runBpOnDevice.mem_management_bp_run->AllocateAlignedMemoryOnDevice(total_dataBottomLevel, ACCELERATION);
    if (runBpOnDevice.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) { return {}; }
  }

  //run bp processing for specified number of runs
  for (unsigned int num_run = 0; num_run < beliefprop::kNumBpStereoRuns; num_run++)
  {
    //allocate the device memory to store and x and y smoothed images
    std::array<float*, 2> smoothed_images{
      runBpOnDevice.mem_management_images->AllocateMemoryOnDevice(totNumPixelsImages),
      runBpOnDevice.mem_management_images->AllocateMemoryOnDevice(totNumPixelsImages)};
    
    if (runBpOnDevice.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      //return std::optional object with no value indicating error in run
      return {};
    }

    //set start timer for specified runtime segments at time before smoothing images
    runtime_start_end_timings[beliefprop::Runtime_Type::kSmoothing][0] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalNoTransfer][0] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalWithTransfer][0] = std::chrono::system_clock::now();

    //first smooth the images using the Gaussian filter with the given smoothing sigma value
    //smoothed images are stored on the target device
    for (unsigned int i = 0; i < 2u; i++) {
      (*(runBpOnDevice.smooth_image))(inputImages[i], alg_settings.smoothing_sigma, smoothed_images[i]);
      if (runBpOnDevice.run_bp_stereo->ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) { 
        //return std::optional object with no value indicating error in run
        return {};
      }
    }

    //end timer for image smoothing and add to image smoothing timings
    runtime_start_end_timings[beliefprop::Runtime_Type::kSmoothing][1] = std::chrono::system_clock::now();

    //get and store time point at start of bp processing
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalBp][0] = std::chrono::system_clock::now();

    //run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object run_bp_stereo
    //returns detailed timings for bp run
    auto rpBpStereoOutput = (*(runBpOnDevice.run_bp_stereo))(
      smoothed_images, alg_settings, width_height_images,
      bpData, bpProcStore, runBpOnDevice.mem_management_bp_run);
    if (!rpBpStereoOutput) {
      //return std::optional object with no value if null output indicating error in run
      return {};
    }

    //get and store end timepoint of bp run for computation of total runtime
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalBp][1] =
      std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalNoTransfer][1] =
      std::chrono::system_clock::now();

    //transfer the disparity map estimation on the device to the host for output
    runBpOnDevice.mem_management_images->TransferDataFromDeviceToHost(
      output_disparity_map.PointerToPixelsStart(), rpBpStereoOutput->first, totNumPixelsImages);

    //compute timings for each portion of interest and add to vector timings
    runtime_start_end_timings[beliefprop::Runtime_Type::kTotalWithTransfer][1] =
      std::chrono::system_clock::now();

    //retrieve the duration for each segment and add to bp timings
    std::ranges::for_each(runtime_start_end_timings,
      [&detailedBPTimings] (const auto& current_runtime_name_timing) {
        detailedBPTimings.AddTiming(current_runtime_name_timing.first,
          current_runtime_name_timing.second[1] - current_runtime_name_timing.second[0]);
    });

    //add bp timings from current run to overall timings
    detailedBPTimings.AddToCurrentTimings(rpBpStereoOutput->second);

    //free the space allocated to the resulting disparity map and smoothed images on the computation device
    runBpOnDevice.mem_management_images->FreeMemoryOnDevice(rpBpStereoOutput->first);
    for (auto& smoothed_image : smoothed_images) {
      runBpOnDevice.mem_management_images->FreeMemoryOnDevice(smoothed_image);
    }
  }

  //free data for bp processing on target device if this memory
  //management set to be done outside of runs
  if constexpr (beliefprop::kAllocateFreeBpMemoryOutsideRuns) {
    runBpOnDevice.mem_management_bp_run->FreeAlignedMemoryOnDevice(bpData);
    runBpOnDevice.mem_management_bp_run->FreeAlignedMemoryOnDevice(bpProcStore);
  }

  //construct RunData object with bp input and timing info
  RunData run_data;
  run_data.AddDataWHeader(std::string(beliefprop::kImageWidthHeader), width_height_images[0]);
  run_data.AddDataWHeader(std::string(beliefprop::kImageHeightHeader), width_height_images[1]);
  run_data.AppendData(detailedBPTimings.AsRunData());

  //construct and return ProcessStereoSetOutput object inside of std::optional object
  return {ProcessStereoSetOutput{
    detailedBPTimings.MedianTiming(
      beliefprop::Runtime_Type::kTotalWithTransfer), std::move(output_disparity_map), run_data}};
}

#endif /* RUNBPSTEREOSET_H_ */
