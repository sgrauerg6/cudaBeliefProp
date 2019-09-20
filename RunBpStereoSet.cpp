/*
 * RunBpStereoSet.cpp
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#include "RunBpStereoSet.h"

enum Runtime_Type { SMOOTHING, TOTAL_BP, TOTAL_NO_TRANSFER, TOTAL_WITH_TRANSFER };

template<typename T>
ProcessStereoSetOutput RunBpStereoSet<T>::processStereoSet(const char* refImagePath, const char* testImagePath,
	const BPsettings& algSettings, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTargetDevice<T>* runBpStereo, RunBpStereoSetMemoryManagement* runBPMemoryMangement)
{
	const std::map<int, std::string> timingNames = {{SMOOTHING, "Smoothing Runtime"}, {TOTAL_BP, "Total BP Runtime"}, {TOTAL_NO_TRANSFER, "Total Runtime not including data transfer time)"},
			{TOTAL_WITH_TRANSFER, "Total runtime including data transfer time"}};
	bool deleteBPMemoryManagementAtEnd = false;
	if (runBPMemoryMangement == nullptr)
	{
		deleteBPMemoryManagementAtEnd = true;
		runBPMemoryMangement = new RunBpStereoSetMemoryManagement();
	}

	unsigned int heightImages = 0;
	unsigned int widthImages = 0;

	std::map<int, std::vector<double>> timings;
	std::for_each(timingNames.begin(), timingNames.end(), [&timings](std::pair<int, std::string> timingName) { timings[timingName.first] = std::vector<double>(); });
	DetailedTimings* detailedTimingsOverall = nullptr;
	float* dispValsHost;

	printf("Start stereo runs\n");
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

		//get timer at time before smoothing images
		auto timeBeforeSmoothImages = std::chrono::system_clock::now();

		//first smooth the images using the Gaussian filter with the given SIGMA_BP value
		//smoothed images are stored on the target device at locations smoothedImage1 and smoothedImage2
		(*smoothImage)(image1AsUnsignedIntArrayHost,
			widthImages, heightImages, algSettings.smoothingSigma, smoothedImage1);
		(*smoothImage)(image2AsUnsignedIntArrayHost,
			widthImages, heightImages, algSettings.smoothingSigma, smoothedImage2);

		//end timer for image smoothing and add to image smoothing timings
		auto endSmoothingTime = std::chrono::system_clock::now();

		//free the host memory allocated to original image 1 and image 2 on the host
		delete[] image1AsUnsignedIntArrayHost;
		delete[] image2AsUnsignedIntArrayHost;

		float* disparityMapFromImage1To2CompDevice;

		//allocate the space for the disparity map estimation
		runBPMemoryMangement->allocateDataOnCompDevice((void**)& disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		//start bp processing time timer
		auto bpTotalTimeStart = std::chrono::system_clock::now();

		//run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
		DetailedTimings* currentDetailedTimings = (*runBpStereo)(smoothedImage1, smoothedImage2,
			disparityMapFromImage1To2CompDevice, algSettings, widthImages, heightImages);

		auto timeAfterBpDoneBeforeTransfer = std::chrono::system_clock::now();

		//transfer the disparity map estimation on the device to the host for output
		runBPMemoryMangement->transferDataFromCompDeviceToHost(dispValsHost, disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		auto timeAfterBpDoneWithTransfer = std::chrono::system_clock::now();

		//compute timings for each portion of interest and add to vector timings
		using timingInSecondsDoublePrecision = std::chrono::duration<double>;
		timings[SMOOTHING].push_back((timingInSecondsDoublePrecision(endSmoothingTime - timeBeforeSmoothImages)).count());
		timings[TOTAL_BP].push_back((timingInSecondsDoublePrecision(timeAfterBpDoneBeforeTransfer - bpTotalTimeStart)).count());
		timings[TOTAL_NO_TRANSFER].push_back((timingInSecondsDoublePrecision(timeAfterBpDoneBeforeTransfer - timeBeforeSmoothImages)).count());
		timings[TOTAL_WITH_TRANSFER].push_back((timingInSecondsDoublePrecision(timeAfterBpDoneWithTransfer - timeBeforeSmoothImages)).count());

		//free memory for disparity if not in last run (where used to create disparity map)
		if (numRun < (NUM_BP_STEREO_RUNS - 1))
		{
			delete [] dispValsHost;
		}

		//free the space allocated to the resulting disparity map and smoothed images on the computation device
		runBPMemoryMangement->freeDataOnCompDevice((void**)& disparityMapFromImage1To2CompDevice);
		runBPMemoryMangement->freeDataOnCompDevice((void**)& smoothedImage1);
		runBPMemoryMangement->freeDataOnCompDevice((void**)& smoothedImage2);

		//check if detailed timings are returned; if so, add to overall detailed timings
		if (currentDetailedTimings != nullptr)
		{
			if (detailedTimingsOverall == nullptr)
			{
				detailedTimingsOverall = currentDetailedTimings;
			}
			else
			{
				//add timings for run to overall set
				detailedTimingsOverall->addTimings(currentDetailedTimings);
			}
		}
	}

	printf("End stereo runs\n");

	//generate output disparity map object
	DisparityMap<float> output_disparity_map(widthImages, heightImages, dispValsHost);
	delete [] dispValsHost;

	fprintf(resultsFile, "Image Width: %d\n", widthImages);
	fprintf(resultsFile, "Image Height: %d\n", heightImages);

	std::for_each(timings.begin(), timings.end(),
			[&timingNames, resultsFile](std::pair<int, std::vector<double>> currentTiming)
			{
				std::sort(currentTiming.second.begin(), currentTiming.second.end());
				fprintf(resultsFile, "Median %s: %f\n", timingNames.at(currentTiming.first).c_str(), currentTiming.second.at(NUM_BP_STEREO_RUNS / 2));
			});

	if (detailedTimingsOverall != nullptr)
	{
		//uncomment to print timings for each part of implementation
		//timings.PrintMedianTimings();
		detailedTimingsOverall->PrintMedianTimingsToFile(resultsFile);
	}

	if (deleteBPMemoryManagementAtEnd)
	{
		delete runBPMemoryMangement;
	}

	ProcessStereoSetOutput output;
	output.runTime = timings[TOTAL_WITH_TRANSFER].at(NUM_BP_STEREO_RUNS / 2);
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
