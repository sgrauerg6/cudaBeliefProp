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
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpConstsAndParams/bpStructsAndEnums.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "BpImageProcessing/BpImage.h"
#include "BpImageProcessing/DisparityMap.h"
#include "BpImageProcessing/SmoothImage.h"
#include "RuntimeTiming/DetailedTimings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "ProcessBPOnTargetDevice.h"
#include "RunBpStereoSetMemoryManagement.h"

//stereo processing output
struct ProcessStereoSetOutput
{
  float runTime = 0.0;
  DisparityMap<float> outDisparityMap;
  RunData runData;
};

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
struct BpOnDevice {
  const std::unique_ptr<SmoothImage>& smoothImage;
  const std::unique_ptr<ProcessBPOnTargetDevice<T, DISP_VALS, ACCELERATION>>& runBpStereo;
  const std::unique_ptr<RunBpStereoSetMemoryManagement<T>>& memManagementBpRun;
  const std::unique_ptr<RunBpStereoSetMemoryManagement<float>>& memManagementImages;
};

template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpStereoSet {
public:
  virtual std::string getBpRunDescription() = 0;

  //pure abstract overloaded operator that must be defined in child class
  virtual ProcessStereoSetOutput operator()(const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings, 
    const beliefprop::ParallelParameters& parallelParams) = 0;

protected:

  //protected function to run stereo processing on any available architecture using pointers to architecture-specific smooth image, process BP, and memory management child class objects
  //using V and W template parameters in default parameter with make_unique works in g++ but not visual studio
  ProcessStereoSetOutput processStereoSet(const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings, const BpOnDevice<T, DISP_VALS, ACCELERATION>& runBpOnDevice);
};


template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
ProcessStereoSetOutput RunBpStereoSet<T, DISP_VALS, ACCELERATION>::processStereoSet(const std::array<std::string, 2>& refTestImagePath,
  const beliefprop::BPsettings& algSettings, const BpOnDevice<T, DISP_VALS, ACCELERATION>& runBpOnDevice)
{
  //retrieve the images as well as the width and height
  const std::array<BpImage<unsigned int>, 2> inputImages{BpImage<unsigned int>(refTestImagePath[0]), BpImage<unsigned int>(refTestImagePath[1])};
  const std::array<unsigned int, 2> widthHeightImages{inputImages[0].getWidth(), inputImages[0].getHeight()};

  //get total number of pixels in input images
  const unsigned int totNumPixelsImages{widthHeightImages[0] * widthHeightImages[1]};

  std::unordered_map<beliefprop::Runtime_Type, std::pair<timingType, timingType>> runtime_start_end_timings;
  DetailedTimings detailedBPTimings(beliefprop::timingNames);

  //generate output disparity map object
  DisparityMap<float> output_disparity_map(widthHeightImages);

  //allocate data for bp processing on target device ahead of runs if option selected
  T* bpData = nullptr;
  T* bpProcStore = nullptr;
  if constexpr (run_environment::ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
    unsigned long numData = beliefprop::levelProperties::getTotalDataForAlignedMemoryAllLevels<T, ACCELERATION>(
      widthHeightImages, algSettings.numDispVals_, algSettings.numLevels_);
    bpData = runBpOnDevice.memManagementBpRun->allocateAlignedMemoryOnDevice(10u*numData, ACCELERATION);
    if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { return {0.0, DisparityMap<float>()}; }

    beliefprop::levelProperties bottomLevelProperties(widthHeightImages, 0, 0, ACCELERATION);
    unsigned long totalDataBottomLevel = bottomLevelProperties.getNumDataInBpArrays<T>(algSettings.numDispVals_);
    bpProcStore = runBpOnDevice.memManagementBpRun->allocateAlignedMemoryOnDevice(totalDataBottomLevel, ACCELERATION);
    if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { return {0.0, DisparityMap<float>()}; }
  }

  for (unsigned int numRun = 0; numRun < bp_params::NUM_BP_STEREO_RUNS; numRun++)
  {
    //allocate the device memory to store and x and y smoothed images
    std::array<float*, 2> smoothedImages{
      runBpOnDevice.memManagementImages->allocateMemoryOnDevice(totNumPixelsImages),
      runBpOnDevice.memManagementImages->allocateMemoryOnDevice(totNumPixelsImages)};
    
    if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { 
      return {0.0, DisparityMap<float>()};
    }

    //set start timer for specified runtime segments at time before smoothing images
    runtime_start_end_timings[beliefprop::Runtime_Type::SMOOTHING].first = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_NO_TRANSFER].first = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_WITH_TRANSFER].first = std::chrono::system_clock::now();

    //first smooth the images using the Gaussian filter with the given smoothing sigma value
    //smoothed images are stored on the target device
    for (unsigned int i = 0; i < 2u; i++) {
      (*(runBpOnDevice.smoothImage))(inputImages[i], algSettings.smoothingSigma_, smoothedImages[i]);
      if (runBpOnDevice.runBpStereo->errorCheck(__FILE__, __LINE__) != run_eval::Status::NO_ERROR) { 
        return {0.0, DisparityMap<float>()};
      }
    }

    //end timer for image smoothing and add to image smoothing timings
    runtime_start_end_timings[beliefprop::Runtime_Type::SMOOTHING].second = std::chrono::system_clock::now();

    //get runtime before bp processing
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_BP].first = std::chrono::system_clock::now();

    //run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
    //returns detailed timings for bp run
    auto rpBpStereoOutput = (*(runBpOnDevice.runBpStereo))(smoothedImages, algSettings, widthHeightImages,
      bpData, bpProcStore, runBpOnDevice.memManagementBpRun);
    if (rpBpStereoOutput.first == nullptr) {
      return {0.0, DisparityMap<float>()};
    }

    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_BP].second = std::chrono::system_clock::now();
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_NO_TRANSFER].second = std::chrono::system_clock::now();

    //transfer the disparity map estimation on the device to the host for output
    runBpOnDevice.memManagementImages->transferDataFromDeviceToHost(
      output_disparity_map.getPointerToPixelsStart(), rpBpStereoOutput.first, totNumPixelsImages);

    //compute timings for each portion of interest and add to vector timings
    runtime_start_end_timings[beliefprop::Runtime_Type::TOTAL_WITH_TRANSFER].second = std::chrono::system_clock::now();

    //retrieve the timing for each runtime segment and add to vector in timings map
    std::for_each(runtime_start_end_timings.begin(), runtime_start_end_timings.end(),
      [&detailedBPTimings] (const auto& currentRuntimeNameAndTiming) {
        detailedBPTimings.addTiming(currentRuntimeNameAndTiming.first,
          (timingInSecondsDoublePrecision(currentRuntimeNameAndTiming.second.second - currentRuntimeNameAndTiming.second.first)).count());
    });

    //add bp timings from current run to overall timings
    detailedBPTimings.addToCurrentTimings(rpBpStereoOutput.second);

    //free the space allocated to the resulting disparity map and smoothed images on the computation device
    runBpOnDevice.memManagementImages->freeMemoryOnDevice(rpBpStereoOutput.first);
    for (auto& smoothedImage : smoothedImages) {
      runBpOnDevice.memManagementImages->freeMemoryOnDevice(smoothedImage);
    }
  }

  //free data for bp processing on target device if this memory
  //management set to be done outside of runs
  if constexpr (run_environment::ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
    runBpOnDevice.memManagementBpRun->freeAlignedMemoryOnDevice(bpData);
    runBpOnDevice.memManagementBpRun->freeAlignedMemoryOnDevice(bpProcStore);
  }

  RunData runData;
  runData.addDataWHeader("Image Width", std::to_string(widthHeightImages[0]));
  runData.addDataWHeader("Image Height", std::to_string(widthHeightImages[1]));
  runData.appendData(detailedBPTimings.runData());

  //construct and return ProcessStereoSetOutput object
  return {(float)detailedBPTimings.getMedianTiming(beliefprop::Runtime_Type::TOTAL_WITH_TRANSFER), std::move(output_disparity_map), runData};
}

#endif /* RUNBPSTEREOSET_H_ */
