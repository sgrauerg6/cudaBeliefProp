/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include <cstring>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <array>
#include <string>
#include "SmoothImage.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include "../RuntimeTiming/DetailedTimings.h"
#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include "../OutputEvaluation/DisparityMap.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"
#include "../RuntimeTiming/DetailedTimings.h"
#include "../ImageDataAndProcessing/BpImage.h"

//stereo processing output
struct ProcessStereoSetOutput
{
	float runTime = 0.0;
	DisparityMap<float> outDisparityMap;
};

template <typename T, unsigned int DISP_VALS>
class RunBpStereoSet {
public:

	virtual ~RunBpStereoSet() {}

	virtual std::string getBpRunDescription() = 0;

	//pure abstract overloaded operator that must be defined in child class
	virtual ProcessStereoSetOutput operator()(const std::array<std::string, 2>& refTestImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream) = 0;

protected:

	//protected function to run stereo processing on any available architecture using pointers to architecture-specific smooth image, process BP, and memory management child class objects
	//using V and W template parameters in default parameter with make_unique works in g++ but not visual studio
	template <typename U, typename V, typename W=float, typename X = float*>
	ProcessStereoSetOutput processStereoSet(const std::array<std::string, 2>& refTestImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream, const std::unique_ptr<SmoothImage<X>>& smoothImage,
		const std::unique_ptr<ProcessBPOnTargetDevice<U, V, DISP_VALS>>& runBpStereo,
		const std::unique_ptr<RunBpStereoSetMemoryManagement<W, X>>& runBPMemoryMangement = 
		std::make_unique<RunBpStereoSetMemoryManagement<
		#ifdef _WIN32
		float, float*
		#else
		W, X
		#endif
		>>() );

};


template<typename T, unsigned int DISP_VALS>
template<typename U, typename V, typename W, typename X>
ProcessStereoSetOutput RunBpStereoSet<T, DISP_VALS>::processStereoSet(const std::array<std::string, 2>& refTestImagePath,
	const BPsettings& algSettings, std::ostream& resultsStream, const std::unique_ptr<SmoothImage<X>>& smoothImage,
	const std::unique_ptr<ProcessBPOnTargetDevice<U, V, DISP_VALS>>& runBpStereo,
	const std::unique_ptr<RunBpStereoSetMemoryManagement<W, X>>& runBPMemoryMangement)
{
	//retrieve the images as well as the width and height
	const std::array<BpImage<unsigned int>, 2> inputImages{BpImage<unsigned int>(refTestImagePath[0]), BpImage<unsigned int>(refTestImagePath[1])};
	const std::array<unsigned int, 2> widthHeightImages{inputImages[0].getWidth(), inputImages[0].getHeight()};

	//get total number of pixels in input images
	const unsigned int totNumPixelsImages{widthHeightImages[0] * widthHeightImages[1]};

	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> runtime_start_end_timings;
	DetailedTimings<Runtime_Type_BP> detailedBPTimings(timingNames_BP);

	//generate output disparity map object
	DisparityMap<float> output_disparity_map(widthHeightImages);

	//allocate data for bp processing on target device ahead of runs if option selected
	V bpData = nullptr;
	void* bpProcStore = nullptr;
	if constexpr (ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
		unsigned long numData = levelProperties::getTotalDataForAlignedMemoryAllLevels<U>(
				widthHeightImages, algSettings.numDispVals_, algSettings.numLevels_);
		bpData = runBpStereo->allocateMemoryOnTargetDevice(10u*numData);

		levelProperties bottomLevelProperties(widthHeightImages);
		unsigned long totalDataBottomLevel = bottomLevelProperties.getNumDataInBpArrays<U>(algSettings.numDispVals_);
		bpProcStore = runBpStereo->allocateMemoryOnTargetDevice(totalDataBottomLevel);
	}

	for (unsigned int numRun = 0; numRun < bp_params::NUM_BP_STEREO_RUNS; numRun++)
	{
		//allocate the device memory to store and x and y smoothed images
		std::array<X, 2> smoothedImages{
			runBPMemoryMangement->allocateDataOnCompDevice(totNumPixelsImages),
			runBPMemoryMangement->allocateDataOnCompDevice(totNumPixelsImages)};

		//set start timer for specified runtime segments at time before smoothing images
		runtime_start_end_timings[Runtime_Type_BP::SMOOTHING].first = std::chrono::system_clock::now();
		runtime_start_end_timings[Runtime_Type_BP::TOTAL_NO_TRANSFER].first = std::chrono::system_clock::now();
		runtime_start_end_timings[Runtime_Type_BP::TOTAL_WITH_TRANSFER].first = std::chrono::system_clock::now();

		//first smooth the images using the Gaussian filter with the given SIGMA_BP value
		//smoothed images are stored on the target device at locations smoothedImage1 and smoothedImage2
		for (unsigned int i = 0; i < 2u; i++) {
			(*smoothImage)(inputImages[i], algSettings.smoothingSigma_, smoothedImages[i]);
		}

		//end timer for image smoothing and add to image smoothing timings
		runtime_start_end_timings[Runtime_Type_BP::SMOOTHING].second = std::chrono::system_clock::now();

		//get runtime before bp processing
		runtime_start_end_timings[Runtime_Type_BP::TOTAL_BP].first = std::chrono::system_clock::now();

		//run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
		//returns detailed timings for bp run
		auto rpBpStereoOutput = (*runBpStereo)(smoothedImages, algSettings, widthHeightImages, bpData, bpProcStore);

		runtime_start_end_timings[Runtime_Type_BP::TOTAL_BP].second = std::chrono::system_clock::now();
		runtime_start_end_timings[Runtime_Type_BP::TOTAL_NO_TRANSFER].second = std::chrono::system_clock::now();

		//transfer the disparity map estimation on the device to the host for output
		runBPMemoryMangement->transferDataFromCompDeviceToHost(
				output_disparity_map.getPointerToPixelsStart(), rpBpStereoOutput.first, totNumPixelsImages);

		//compute timings for each portion of interest and add to vector timings
		runtime_start_end_timings[Runtime_Type_BP::TOTAL_WITH_TRANSFER].second = std::chrono::system_clock::now();

		//retrieve the timing for each runtime segment and add to vector in timings map
		std::for_each(runtime_start_end_timings.begin(), runtime_start_end_timings.end(),
				[&detailedBPTimings] (const auto& currentRuntimeNameAndTiming) {
			detailedBPTimings.addTiming(currentRuntimeNameAndTiming.first,
					(timingInSecondsDoublePrecision(currentRuntimeNameAndTiming.second.second - currentRuntimeNameAndTiming.second.first)).count());
		});

		//add bp timings from current run to overall timings
		detailedBPTimings.addToCurrentTimings(rpBpStereoOutput.second);

		//free the space allocated to the resulting disparity map and smoothed images on the computation device
		runBPMemoryMangement->freeDataOnCompDevice(rpBpStereoOutput.first);
		for (auto& smoothedImage : smoothedImages) {
			runBPMemoryMangement->freeDataOnCompDevice(smoothedImage);
		}
	}

	//free data for bp processing on target device if this memory
	//management set to be done outside of runs
	if constexpr (ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS) {
		runBpStereo->freeMemoryOnTargetDevice(bpData);
		runBpStereo->freeRawMemoryOnTargetDevice(bpProcStore);
	}

	resultsStream << "Image Width: " << widthHeightImages[0] << "\nImage Height: " << widthHeightImages[1] << "\n";
	resultsStream << detailedBPTimings;

	//construct and return ProcessStereoSetOutput object
	return {(float)detailedBPTimings.getMedianTiming(Runtime_Type_BP::TOTAL_WITH_TRANSFER), std::move(output_disparity_map)};
}

#endif /* RUNBPSTEREOSET_H_ */
