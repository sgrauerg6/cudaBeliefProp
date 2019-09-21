/*
 * RunBpStereoSet.cpp
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#include "RunBpStereoSet.h"

typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

//std::pair<std::pair<unsigned int, std::string>, double>
template<typename T>
ProcessStereoSetOutput RunBpStereoSet<T>::processStereoSet(const std::string refImagePath, const std::string testImagePath,
	const BPsettings& algSettings, std::ostream& resultsStream, const std::unique_ptr<SmoothImage>& smoothImage, const std::unique_ptr<ProcessBPOnTargetDevice<T>>& runBpStereo,
	const std::unique_ptr<RunBpStereoSetMemoryManagement>& runBPMemoryMangement)
{
	unsigned int heightImages = 0;
	unsigned int widthImages = 0;

	//retrieve the images as well as the width and height
	unsigned int* image1AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
				refImagePath, widthImages, heightImages);
	unsigned int* image2AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
				testImagePath, widthImages, heightImages);

	std::unordered_map<Runtime_Type_BP, std::pair<timingType, timingType>> runtime_start_end_timings;
	DetailedTimings<Runtime_Type_BP> detailedBPTimings(timingNames_BP);

	//generate output disparity map object
	DisparityMap<float> output_disparity_map(widthImages, heightImages);

	//get shared pointer to disparity map data
	//std::unique_ptr<float[]> dispValsSharedPtr = std::move(output_disparity_map.getDisparityValuesSharedPointer());

	for (int numRun = 0; numRun < NUM_BP_STEREO_RUNS; numRun++)
	{
		float* smoothedImage1;
		float* smoothedImage2;

		//allocate the device memory to store and x and y smoothed images
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& smoothedImage1, widthImages * heightImages * sizeof(float));
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& smoothedImage2, widthImages * heightImages * sizeof(float));

		//set start timer for specified runtime segments at time before smoothing images
		runtime_start_end_timings[SMOOTHING].first = runtime_start_end_timings[TOTAL_NO_TRANSFER].first = runtime_start_end_timings[TOTAL_WITH_TRANSFER].first = std::chrono::system_clock::now();

		//first smooth the images using the Gaussian filter with the given SIGMA_BP value
		//smoothed images are stored on the target device at locations smoothedImage1 and smoothedImage2
		(*smoothImage)(image1AsUnsignedIntArrayHost,
			widthImages, heightImages, algSettings.smoothingSigma, smoothedImage1);
		(*smoothImage)(image2AsUnsignedIntArrayHost,
			widthImages, heightImages, algSettings.smoothingSigma, smoothedImage2);

		//end timer for image smoothing and add to image smoothing timings
		runtime_start_end_timings[SMOOTHING].second = std::chrono::system_clock::now();

		float* disparityMapFromImage1To2CompDevice;

		//allocate the space for the disparity map estimation
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		//get runtime before bp processing
		runtime_start_end_timings[TOTAL_BP].first = std::chrono::system_clock::now();

		//run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
		//returns detailed timings for bp run
		DetailedTimings<Runtime_Type_BP> currentDetailedTimings = (*runBpStereo)(smoothedImage1, smoothedImage2,
			disparityMapFromImage1To2CompDevice, algSettings, widthImages, heightImages);

		runtime_start_end_timings[TOTAL_BP].second = runtime_start_end_timings[TOTAL_NO_TRANSFER].second = std::chrono::system_clock::now();

		//transfer the disparity map estimation on the device to the host for output
		runBPMemoryMangement->transferDataFromCompDeviceToHost(&(output_disparity_map.getDisparityValuesUniquePointer()[0]), disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

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
		runBPMemoryMangement->freeDataOnCompDevice((void**)& disparityMapFromImage1To2CompDevice);
		runBPMemoryMangement->freeDataOnCompDevice((void**)& smoothedImage1);
		runBPMemoryMangement->freeDataOnCompDevice((void**)& smoothedImage2);
	}

	//free the host memory allocated to original image 1 and image 2 on the host
	delete[] image1AsUnsignedIntArrayHost;
	delete[] image2AsUnsignedIntArrayHost;

	resultsStream << "Image Width: " << widthImages << "\nImage Height: " << heightImages << "\n";
	resultsStream << detailedBPTimings;

	ProcessStereoSetOutput output;
	output.runTime = detailedBPTimings.getMedianTiming(TOTAL_WITH_TRANSFER);
	output.outDisparityMap = std::move(output_disparity_map);

	return output;
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class RunBpStereoSet<float>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class RunBpStereoSet<double>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class RunBpStereoSet<float16_t>;
#else
template class RunBpStereoSet<short>;
#endif //COMPILING_FOR_ARM

#endif
