/*
 * RunBpStereoSet.cpp
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#include "RunBpStereoSet.h"
#include <unordered_map>
#include "DetailedTimings.h"

typedef std::chrono::time_point<std::chrono::system_clock> timingType;
using timingInSecondsDoublePrecision = std::chrono::duration<double>;

enum Runtime_Type { SMOOTHING, TOTAL_BP, TOTAL_NO_TRANSFER, TOTAL_WITH_TRANSFER };
const std::unordered_map<unsigned int, std::string> timingNames = {{SMOOTHING, "Smoothing Runtime"}, {TOTAL_BP, "Total BP Runtime"}, {TOTAL_NO_TRANSFER, "Total Runtime not including data transfer time)"},
			{TOTAL_WITH_TRANSFER, "Total runtime including data transfer time"}};

//std::pair<std::pair<unsigned int, std::string>, double>
template<typename T>
ProcessStereoSetOutput RunBpStereoSet<T>::processStereoSet(const char* refImagePath, const char* testImagePath,
	const BPsettings& algSettings, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTargetDevice<T>* runBpStereo, RunBpStereoSetMemoryManagement* runBPMemoryMangement)
{
	bool deleteBPMemoryManagementAtEnd = false;
	if (runBPMemoryMangement == nullptr)
	{
		deleteBPMemoryManagementAtEnd = true;
		runBPMemoryMangement = new RunBpStereoSetMemoryManagement();
	}

	unsigned int heightImages = 0;
	unsigned int widthImages = 0;

	std::unordered_map<unsigned int, std::pair<timingType, timingType>> runtime_start_end_timings;

	DetailedTimings detailedTimingsOverall;
	float* dispValsHost;
	DetailedTimings segmentTimings;

	for (int numRun = 0; numRun < NUM_BP_STEREO_RUNS; numRun++)
	{
		//first run Stereo estimation on the first two images
		unsigned int* image1AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
			refImagePath, widthImages, heightImages);
		unsigned int* image2AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
			testImagePath, widthImages, heightImages);

		float* smoothedImage1;
		float* smoothedImage2;

		//allocate the device memory to store and x and y smoothed images
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& smoothedImage1, widthImages * heightImages * sizeof(float));
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& smoothedImage2, widthImages * heightImages * sizeof(float));
		dispValsHost = new float[widthImages*heightImages];

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

		//free the host memory allocated to original image 1 and image 2 on the host
		delete[] image1AsUnsignedIntArrayHost;
		delete[] image2AsUnsignedIntArrayHost;

		float* disparityMapFromImage1To2CompDevice;

		//allocate the space for the disparity map estimation
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		//get runtime before bp processing
		runtime_start_end_timings[TOTAL_BP].first = std::chrono::system_clock::now();

		//run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
		DetailedTimings currentDetailedTimings = (*runBpStereo)(smoothedImage1, smoothedImage2,
			disparityMapFromImage1To2CompDevice, algSettings, widthImages, heightImages);

		runtime_start_end_timings[TOTAL_BP].second = runtime_start_end_timings[TOTAL_NO_TRANSFER].second = std::chrono::system_clock::now();

		//transfer the disparity map estimation on the device to the host for output
		runBPMemoryMangement->transferDataFromCompDeviceToHost(dispValsHost, disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		//compute timings for each portion of interest and add to vector timings
		runtime_start_end_timings[TOTAL_WITH_TRANSFER].second = std::chrono::system_clock::now();

		//retrieve the timing for each runtime segment and add to vector in timings map
		std::for_each(timingNames.begin(), timingNames.end(),
				[&runtime_start_end_timings, &segmentTimings] (auto& currentRuntimeSegmentAndName) {
			segmentTimings.addTiming(std::make_pair(currentRuntimeSegmentAndName,
					(timingInSecondsDoublePrecision(runtime_start_end_timings[currentRuntimeSegmentAndName.first].second - runtime_start_end_timings[currentRuntimeSegmentAndName.first].first)).count()));
		});

		//free memory for disparity if not in last run (where used to create disparity map)
		if (numRun < (NUM_BP_STEREO_RUNS - 1))
		{
			delete [] dispValsHost;
		}

		//free the space allocated to the resulting disparity map and smoothed images on the computation device
		runBPMemoryMangement->freeDataOnCompDevice((void**)& disparityMapFromImage1To2CompDevice);
		runBPMemoryMangement->freeDataOnCompDevice((void**)& smoothedImage1);
		runBPMemoryMangement->freeDataOnCompDevice((void**)& smoothedImage2);

		detailedTimingsOverall.addToCurrentTimings(currentDetailedTimings);
	}

	//generate output disparity map object
	DisparityMap<float> output_disparity_map(widthImages, heightImages, dispValsHost);
	delete [] dispValsHost;

	fprintf(resultsFile, "Image Width: %d\nImage Height: %d\n", widthImages, heightImages);
	segmentTimings.printCurrentTimings();
	detailedTimingsOverall.printCurrentTimings();

	if (deleteBPMemoryManagementAtEnd)
	{
		delete runBPMemoryMangement;
	}

	ProcessStereoSetOutput output;
	output.runTime = segmentTimings.getMedianTiming(TOTAL_WITH_TRANSFER);
	output.outDisparityMap = output_disparity_map;

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
