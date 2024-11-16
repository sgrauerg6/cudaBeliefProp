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
#include "BpConstsAndParams/BpConsts.h"
#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "BpImageProcessing/BpImage.h"
#include "BpImageProcessing/DisparityMap.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "RunImp/RunImpMemoryManagement.h"
#include "ProcessBPOnTargetDevice.h"

//structure with output disparity map, runtime, and other evaluation data
struct ProcessStereoSetOutput
{
  std::chrono::duration<double> runTime;
  DisparityMap<float> outDisparityMap;
  RunData runData;
};

//structure with pointers to objects containing functions for smoothing images,
//processing bp, and memory management on target device using specified acceleration
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
struct BpOnDevice {
  const std::unique_ptr<SmoothImage>& smoothImage;
  const std::unique_ptr<ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>>& runBpStereo;
  const std::unique_ptr<RunImpMemoryManagement<T>>& memManagementBpRun;
  const std::unique_ptr<RunImpMemoryManagement<float>>& memManagementImages;
};

//abstract class to set up and run belief propagation on target device using specified acceleration
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpStereoSet {
public:
  //pure virtual function to return run description corresponding to target acceleration
  virtual std::string getBpRunDescription() const = 0;

  //pure virtual operator() overload that must be defined in child class
  virtual std::optional<ProcessStereoSetOutput> operator()(
    const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings, 
    const ParallelParams& parallelParams) = 0;

protected:
  //protected function to set up, run, and evaluate bp processing on target device using pointers to acceleration-specific smooth image,
  //process BP, and memory management child class objects
  std::optional<ProcessStereoSetOutput> processStereoSet(
    const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings,
    const BpOnDevice<T, DISP_VALS, ACCELERATION>& runBpOnDevice);
};

//protected function to set up, run, and evaluate bp processing on target device using pointers to acceleration-specific smooth image,
//process BP, and memory management child class objects
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
std::optional<ProcessStereoSetOutput> RunBpStereoSet<T, DISP_VALS, ACCELERATION>::processStereoSet(
  const std::array<std::string, 2>& refTestImagePath,
  const beliefprop::BPsettings& algSettings,
  const BpOnDevice<T, DISP_VALS, ACCELERATION>& runBpOnDevice)
{
  //retrieve the images as well as the width and height
  const std::array<BpImage<unsigned int>, 2> inputImages{BpImage<unsigned int>(refTestImagePath[0]), BpImage<unsigned int>(refTestImagePath[1])};
  const std::array<unsigned int, 2> widthHeightImages{inputImages[0].getWidth(), inputImages[0].getHeight()};

  //get total number of pixels in input images
  const unsigned int totNumPixelsImages{widthHeightImages[0] * widthHeightImages[1]};

  //initialize structures for timing data
  std::unordered_map<beliefprop::Runtime_Type, std::array<timingType, 2>> runtime_start_end_timings;
  DetailedTimings detailedBPTimings(beliefprop::timingNames);

  //initialize output disparity map
  DisparityMap<float> output_disparity_map(widthHeightImages);

  //allocate data for bp processing on target device ahead of runs if option selected
  T* bpData{nullptr};
  T* bpProcStore{nullptr};
  if constexpr (bp_params::ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
    //allocate memory on device for bp processing
    const unsigned long numData = beliefprop::levelProperties::getTotalDataForAlignedMemoryAllLevels<T, ACCELERATION>(
      widthHeightImages, algSettings.numDispVals_, algSettings.numLevels_);
    bpData = runBpOnDevice.memManagementBpRun->allocateAlignedMemoryOnDevice(10u*numData, ACCELERATION);
    if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { return {}; }

    beliefprop::levelProperties bottomLevelProperties(widthHeightImages, 0, 0, ACCELERATION);
    const unsigned long totalDataBottomLevel = bottomLevelProperties.getNumDataInBpArrays<T>(algSettings.numDispVals_);
    bpProcStore = runBpOnDevice.memManagementBpRun->allocateAlignedMemoryOnDevice(totalDataBottomLevel, ACCELERATION);
    if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { return {}; }
  }

  //run bp processing for specified number of runs
  for (unsigned int numRun = 0; numRun < bp_params::NUM_BP_STEREO_RUNS; numRun++)
  {
    //allocate the device memory to store and x and y smoothed images
    std::array<float*, 2> smoothedImages{
      runBpOnDevice.memManagementImages->allocateMemoryOnDevice(totNumPixelsImages),
      runBpOnDevice.memManagementImages->allocateMemoryOnDevice(totNumPixelsImages)};
    
    if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) {
      //return std::optional object with no value indicating error in run
      return {};
    }

    //set start timer for specified runtime segments at time before smoothing images
    runtime_start_end_timings[beliefprop::Runtime_Type::SMOOTHING][0] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_NO_TRANSFER][0] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_WITH_TRANSFER][0] = std::chrono::system_clock::now();

    //first smooth the images using the Gaussian filter with the given smoothing sigma value
    //smoothed images are stored on the target device
    for (unsigned int i = 0; i < 2u; i++) {
      (*(runBpOnDevice.smoothImage))(inputImages[i], algSettings.smoothingSigma_, smoothedImages[i]);
      if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { 
        //return std::optional object with no value indicating error in run
        return {};
      }
    }

    //end timer for image smoothing and add to image smoothing timings
    runtime_start_end_timings[beliefprop::Runtime_Type::SMOOTHING][1] = std::chrono::system_clock::now();

    //get and store time point at start of bp processing
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_BP][0] = std::chrono::system_clock::now();

    //run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
    //returns detailed timings for bp run
    auto rpBpStereoOutput = (*(runBpOnDevice.runBpStereo))(smoothedImages, algSettings, widthHeightImages,
      bpData, bpProcStore, runBpOnDevice.memManagementBpRun);
    if (!rpBpStereoOutput) {
      //return std::optional object with no value if null output indicating error in run
      return {};
    }

    //get and store end timepoint of bp run for computation of total runtime
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_BP][1] = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_NO_TRANSFER][1] = std::chrono::system_clock::now();

    //transfer the disparity map estimation on the device to the host for output
    runBpOnDevice.memManagementImages->transferDataFromDeviceToHost(
      output_disparity_map.getPointerToPixelsStart(), rpBpStereoOutput->first, totNumPixelsImages);

    //compute timings for each portion of interest and add to vector timings
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_WITH_TRANSFER][1] = std::chrono::system_clock::now();

    //retrieve the duration for each segment and add to bp timings
    std::ranges::for_each(runtime_start_end_timings,
      [&detailedBPTimings] (const auto& currentRuntimeNameAndTiming) {
        detailedBPTimings.addTiming(currentRuntimeNameAndTiming.first,
          currentRuntimeNameAndTiming.second[1] - currentRuntimeNameAndTiming.second[0]);
    });

    //add bp timings from current run to overall timings
    detailedBPTimings.addToCurrentTimings(rpBpStereoOutput->second);

    //free the space allocated to the resulting disparity map and smoothed images on the computation device
    runBpOnDevice.memManagementImages->freeMemoryOnDevice(rpBpStereoOutput->first);
    for (auto& smoothedImage : smoothedImages) {
      runBpOnDevice.memManagementImages->freeMemoryOnDevice(smoothedImage);
    }
  }

  //free data for bp processing on target device if this memory
  //management set to be done outside of runs
  if constexpr (bp_params::ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
    runBpOnDevice.memManagementBpRun->freeAlignedMemoryOnDevice(bpData);
    runBpOnDevice.memManagementBpRun->freeAlignedMemoryOnDevice(bpProcStore);
  }

  //construct RunData object with bp input and timing info
  RunData runData;
  runData.addDataWHeader(std::string(belief_prop::IMAGE_WIDTH_HEADER), widthHeightImages[0]);
  runData.addDataWHeader(std::string(belief_prop::IMAGE_HEIGHT_HEADER), widthHeightImages[1]);
  runData.appendData(detailedBPTimings.runData());

  //construct and return ProcessStereoSetOutput object inside of std::optional object
  return {ProcessStereoSetOutput{
    detailedBPTimings.getMedianTiming(beliefprop::Runtime_Type::TOTAL_WITH_TRANSFER), std::move(output_disparity_map), runData}};
}

#endif /* RUNBPSTEREOSET_H_ */
