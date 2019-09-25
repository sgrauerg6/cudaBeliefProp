/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include "../ParameterFiles/bpStereoParameters.h"
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStructsAndEnums.h"
#include "SmoothImage.h"
#include "../BpAndSmoothProcessing/ProcessBPOnTargetDevice.h"
#include <cstring>
#include "../RuntimeTiming/DetailedTimings.h"
#include "../BpAndSmoothProcessing/RunBpStereoSetMemoryManagement.h"
#include "../OutputEvaluation/DisparityMap.h"
#include "../RuntimeTiming/DetailedTimingBPConsts.h"
#include <iostream>
#include <unordered_map>
#include "../RuntimeTiming/DetailedTimings.h"
#include <memory>
#include "../ImageDataAndProcessing/BpImage.h"

//stereo processing output
struct ProcessStereoSetOutput
{
	float runTime = 0.0;
	DisparityMap<float> outDisparityMap;
};

template <typename T>
class RunBpStereoSet {
public:
	RunBpStereoSet() {
	}

	virtual ~RunBpStereoSet() {
	}

	virtual std::string getBpRunDescription() = 0;

	//pure abstract overloaded operator that must be defined in child class
	virtual ProcessStereoSetOutput operator()(const std::string& refImagePath, const std::string& testImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream) = 0;

protected:

	//protected function to run stereo processing on any available architecture using pointers to architecture-specific smooth image, process BP, and memory management child class objects
	template <typename U, typename V = float, typename W = float*>
	ProcessStereoSetOutput processStereoSet(const std::string& refImagePath, const std::string& testImagePath,
		const BPsettings& algSettings, std::ostream& resultsStream, const std::unique_ptr<SmoothImage<W>>& smoothImage,
		const std::unique_ptr<ProcessBPOnTargetDevice<T, U>>& runBpStereo, const std::unique_ptr<RunBpStereoSetMemoryManagement<V, W>>& runBPMemoryMangement = std::make_unique<RunBpStereoSetMemoryManagement<V, W>>() );

};



typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

//std::pair<std::pair<unsigned int, std::string>, double>
template<typename T>
template<typename U, typename V, typename W>
ProcessStereoSetOutput RunBpStereoSet<T>::processStereoSet(const std::string& refImagePath, const std::string& testImagePath,
	const BPsettings& algSettings, std::ostream& resultsStream, const std::unique_ptr<SmoothImage<W>>& smoothImage, const std::unique_ptr<ProcessBPOnTargetDevice<T, U>>& runBpStereo,
	const std::unique_ptr<RunBpStereoSetMemoryManagement<V, W>>& runBPMemoryMangement)
{
	//retrieve the images as well as the width and height
	BpImage<unsigned int> image1AsUnsignedIntArrayHost(refImagePath);
	BpImage<unsigned int> image2AsUnsignedIntArrayHost(testImagePath);

	const unsigned int heightImages = image1AsUnsignedIntArrayHost.getHeight();
	const unsigned int widthImages = image1AsUnsignedIntArrayHost.getWidth();

	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> runtime_start_end_timings;
	DetailedTimings<Runtime_Type_BP> detailedBPTimings(timingNames_BP);

	//generate output disparity map object
	DisparityMap<float> output_disparity_map(widthImages, heightImages);

	for (unsigned int numRun = 0; numRun < bp_params::NUM_BP_STEREO_RUNS; numRun++)
	{
		//allocate the device memory to store and x and y smoothed images
		W smoothedImage1 = runBPMemoryMangement->allocateDataOnCompDevice(widthImages * heightImages);
		W smoothedImage2 = runBPMemoryMangement->allocateDataOnCompDevice(widthImages * heightImages);

		//set start timer for specified runtime segments at time before smoothing images
		runtime_start_end_timings[SMOOTHING].first = runtime_start_end_timings[TOTAL_NO_TRANSFER].first = runtime_start_end_timings[TOTAL_WITH_TRANSFER].first = std::chrono::system_clock::now();

		//first smooth the images using the Gaussian filter with the given SIGMA_BP value
		//smoothed images are stored on the target device at locations smoothedImage1 and smoothedImage2
		(*smoothImage)(image1AsUnsignedIntArrayHost, algSettings.smoothingSigma, smoothedImage1);
		(*smoothImage)(image2AsUnsignedIntArrayHost, algSettings.smoothingSigma, smoothedImage2);

		//end timer for image smoothing and add to image smoothing timings
		runtime_start_end_timings[SMOOTHING].second = std::chrono::system_clock::now();

		//allocate the space for the disparity map estimation
		W disparityMapFromImage1To2CompDevice = runBPMemoryMangement->allocateDataOnCompDevice(widthImages * heightImages);

		//get runtime before bp processing
		runtime_start_end_timings[TOTAL_BP].first = std::chrono::system_clock::now();

		//run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
		//returns detailed timings for bp run
		DetailedTimings<Runtime_Type_BP> currentDetailedTimings = (*runBpStereo)(smoothedImage1, smoothedImage2,
			disparityMapFromImage1To2CompDevice, algSettings, widthImages, heightImages);

		runtime_start_end_timings[TOTAL_BP].second = runtime_start_end_timings[TOTAL_NO_TRANSFER].second = std::chrono::system_clock::now();

		//transfer the disparity map estimation on the device to the host for output
		runBPMemoryMangement->transferDataFromCompDeviceToHost(output_disparity_map.getPointerToPixelsStart(), disparityMapFromImage1To2CompDevice, widthImages * heightImages);

		//compute timings for each portion of interest and add to vector timings
		runtime_start_end_timings[TOTAL_WITH_TRANSFER].second = std::chrono::system_clock::now();

		//retrieve the timing for each runtime segment and add to vector in timings map
		std::for_each(runtime_start_end_timings.begin(), runtime_start_end_timings.end(),
				[&detailedBPTimings] (auto& currentRuntimeNameAndTiming) {
			detailedBPTimings.addTiming(currentRuntimeNameAndTiming.first,
					(timingInSecondsDoublePrecision(currentRuntimeNameAndTiming.second.second - currentRuntimeNameAndTiming.second.first)).count());
		});

		//add bp timings from current run to overall timings
		detailedBPTimings.addToCurrentTimings(currentDetailedTimings);

		//free the space allocated to the resulting disparity map and smoothed images on the computation device
		runBPMemoryMangement->freeDataOnCompDevice(disparityMapFromImage1To2CompDevice);
		runBPMemoryMangement->freeDataOnCompDevice(smoothedImage1);
		runBPMemoryMangement->freeDataOnCompDevice(smoothedImage2);
	}

	resultsStream << "Image Width: " << widthImages << "\nImage Height: " << heightImages << "\n";
	resultsStream << detailedBPTimings;

	ProcessStereoSetOutput output;
	output.runTime = detailedBPTimings.getMedianTiming(TOTAL_WITH_TRANSFER);
	output.outDisparityMap = std::move(output_disparity_map);

	return output;
}

#endif /* RUNBPSTEREOSET_H_ */
